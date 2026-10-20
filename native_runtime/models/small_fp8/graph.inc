    // Lightweight FP8 graph recording; included inside phoenix_cpp::Graph.
    // The recovered stage counts are {2,2,2,2,2}, followed by eight transformers.
    // Block IDs below belong to this model, not to the original 71-block model.

    struct SmallFp8WindowShift {
        int x, y;
    };
    static constexpr std::array<SmallFp8WindowShift, 4> small_fp8_window_shifts_{
        {{0, 0}, {-1, -1}, {-1, 0}, {0, -1}}};

    Tensor record_small_fp8_encoder(Tensor features, std::map<int, Tensor>& encoder_skips) {
        struct Stage {
            int first_block, last_block;
        };
        static constexpr std::array<Stage, 4> stages{{{1, 2}, {3, 4}, {5, 6}, {7, 8}}};
        Tensor window_accumulator;

        // Each spatial stage keeps its last full-resolution output for the
        // decoder, then projects the pooled accumulator to twice the channels.
        for (const auto stage : stages) {
            for (int block = stage.first_block; block <= stage.last_block; ++block) {
                current_block_ = block;
                const auto shift = small_fp8_window_shifts_[block - stage.first_block];
                const bool downsample = block == stage.last_block;
                features = shifted_window_block(features, block, shift.x, shift.y, false,
                                                downsample ? &window_accumulator : nullptr);
                if (downsample) {
                    encoder_skips[block] = features;
                    features =
                        matrix_multiply(window_accumulator, model_weight(block, 0, "resample"),
                                        features.channels * 2, 5, true);
                    features = pad_spatial(features, align_extent(features.height, 4),
                                           align_extent(features.width, 4));
                }
                checkpoint("block" + std::to_string(block), features);
            }
        }

        // The 512-channel level separates feed-forward and attention operators.
        // Retain its last output before the final pool into the global tokens.
        constexpr int first_split_block = 9, last_split_block = 10;
        for (int block = first_split_block; block <= last_split_block; ++block) {
            current_block_ = block;
            const auto shift = small_fp8_window_shifts_[block - first_split_block];
            features = split_window_block(features, block, shift.x, shift.y, &window_accumulator);
            checkpoint("block" + std::to_string(block), features);
        }
        encoder_skips[last_split_block] = features;
        auto pooled =
            pad_spatial(average_pool(window_accumulator), align_extent(features.height / 2, 4),
                        align_extent(features.width / 2, 4));
        current_block_ = last_split_block;
        current_layer_ = 4;
        features =
            matrix_multiply(pooled, model_weight(last_split_block, 4, "weight"), 1024, 5, true);
        checkpoint("pooled10", features);
        return features;
    }

    Tensor record_small_fp8_transformers(Tensor features, const std::array<unsigned, 8>& repeats) {
        constexpr int first_transformer_block = 11;
        for (unsigned index = 0; index < repeats.size(); ++index) {
            demand(repeats[index] <= 8, "Transformer repeat count exceeds eight");
            const int block = first_transformer_block + int(index);
            current_block_ = block;
            for (unsigned repetition = 0; repetition < repeats[index]; ++repetition)
                features = transformer_block(features, block);
            checkpoint("block" + std::to_string(block), features);
        }
        return features;
    }

    Tensor record_small_fp8_decoder(Tensor features, std::map<int, Tensor>& encoder_skips) {
        constexpr int decoder_input_block = 19;
        current_block_ = decoder_input_block;
        features = upsample(features, encoder_skips.at(10), decoder_input_block, true);
        checkpoint("block19", features);

        // The recovered decoder rule is (stage_block_count + index) % 4.
        // Every small-model level has two blocks, so all levels use phases 2,3,
        // including the 512-channel split-window level.
        constexpr unsigned decoder_first_phase = 2;
        constexpr int first_split_block = 20, last_split_block = 21;
        for (int block = first_split_block; block <= last_split_block; ++block) {
            current_block_ = block;
            const auto shift =
                small_fp8_window_shifts_[decoder_first_phase + block - first_split_block];
            features = split_window_block(features, block, shift.x, shift.y);
            checkpoint("block" + std::to_string(block), features);
        }

        struct Stage {
            int first_block, encoder_skip_block;
        };
        static constexpr std::array<Stage, 4> stages{{{22, 8}, {24, 6}, {26, 4}, {28, 2}}};
        for (const auto stage : stages) {
            for (int index = 0; index < 2; ++index) {
                const int block = stage.first_block + index;
                current_block_ = block;
                if (index == 0)
                    features = upsample(features, encoder_skips.at(stage.encoder_skip_block), block,
                                        false);
                const auto shift = small_fp8_window_shifts_[decoder_first_phase + index];
                features = shifted_window_block(features, block, shift.x, shift.y);
                checkpoint("block" + std::to_string(block), features);
            }
            encoder_skips.erase(stage.encoder_skip_block);
        }
        return features;
    }

    void record_small_fp8_output(Tensor features, Tensor& stem_skip) {
        constexpr int output_block = 30;
        current_block_ = output_block;
        current_layer_ = 0;
        auto blend_gains = upload_weight(
            "post_sincos",
            [&] {
                auto main_gain = model_.half(output_block, 0, "main_gain");
                auto skip_gain = model_.half(output_block, 0, "skip_gain");
                main_gain.insert(main_gain.end(), skip_gain.begin(), skip_gain.end());
                return main_gain;
            },
            false);
        features = shifted_window_block(features, output_block, -1, -1, true, nullptr, &stem_skip,
                                        &blend_gains);

        // The stored output projection contains more coefficients than this
        // four-channel runtime head consumes. Preserve the exact 4x32 slice.
        output = matrix_multiply(features,
                                 upload_weight("output_head",
                                               [&] {
                                                   auto coefficients =
                                                       model_.half(output_block, 0, "out_project");
                                                   coefficients.resize(4 * 32);
                                                   return coefficients;
                                               }),
                                 4);
        checkpoint("head", output);
    }

    void build_small_fp8(unsigned frame_width, unsigned frame_height,
                         const std::array<unsigned, 8>& repeats) {
        demand(commands.empty(), "Graph is immutable after recording");
        width = frame_width;
        height = frame_height;

        // Lift 16 input features to 32 channels. Keep the stem's full-resolution
        // skip for the head, and start the encoder with its pooled accumulator.
        auto input_owner = allocate_tensor(frame_height, frame_width, 16);
        input = input_owner;
        input.memory = std::make_shared<Allocation>(*input_owner.memory);
        current_block_ = 0;
        Tensor stem_accumulator;
        std::map<int, Tensor> encoder_skips;
        encoder_skips[0] = shifted_window_block(input_owner, 0, 0, 0, true, &stem_accumulator);
        auto features = pad_spatial(stem_accumulator, align_extent(frame_height / 2, 4),
                                    align_extent(frame_width / 2, 4));
        input_owner = {};
        checkpoint("block0", features);
        checkpoint("skip0", encoder_skips.at(0));

        features = record_small_fp8_encoder(features, encoder_skips);
        features = record_small_fp8_transformers(features, repeats);
        features = record_small_fp8_decoder(features, encoder_skips);
        record_small_fp8_output(features, encoder_skips.at(0));

        if (device_features_) {
            // Upload views precede the first neural command and share its scratch
            // arena. The noise view can remain resident when device caching is on.
            frame_planes_[0] = allocate_tensor(height, width, 3, 4);
            frame_planes_[1] = allocate_tensor(height, width, 3, 4);
            frame_planes_[2] = allocate_tensor(height, width, 3, 2);
            // Validity bytes followed by an aligned optional RGBA32F map.
            frame_planes_[3] = allocate_tensor(height, width, 5, 4);
        }
        commit_scratch();
    }
