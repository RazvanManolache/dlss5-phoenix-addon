// Original-model graph recording; included inside phoenix_cpp::Graph.
    void build_original(unsigned frame_width, unsigned frame_height,
               const std::array<unsigned, 8>& repeats) {
        demand(commands.empty(), "Graph is immutable after recording");
        width = frame_width;
        height = frame_height;
        auto input_owner = allocate_tensor(frame_height, frame_width, 16);
        input = input_owner;
        input.memory = std::make_shared<Allocation>(*input_owner.memory);
        current_block_ = 0;

        // Stem: lift the 16 input features to 32 channels, retain the first
        // full-resolution skip, and start the spatial encoder.
        auto features = input_owner;
        Tensor accumulator;
        auto stem = shifted_window_block(features, 0, 0, 0, true, &accumulator);
        std::map<int, Tensor> skips;
        skips[0] = stem;
        features = pad_spatial(accumulator, align_extent(frame_height / 2, 4),
                               align_extent(frame_width / 2, 4));
        input_owner = {};
        checkpoint("block0", features);
        checkpoint("skip0", skips[0]);

        const int shifts[4][2] = {{0, 0}, {-1, -1}, {-1, 0}, {0, -1}};
        int block = 1;
        // The last block at each encoder level also pools its accumulator.
        for (auto level_blocks : {4, 4, 6, 8}) {
            for (int index = 0; index < level_blocks; ++index, ++block) {
                current_block_ = block;
                features = shifted_window_block(features, block, shifts[index % 4][0],
                                                shifts[index % 4][1], false,
                                                index == level_blocks - 1 ? &accumulator : nullptr);
                if (index == level_blocks - 1) {
                    skips[block] = features;
                    auto pooled = accumulator;
                    features = matrix_multiply(pooled, model_weight(block, 0, "resample"),
                                               features.channels * 2, 5, true);
                    features = pad_spatial(features, align_extent(features.height, 4),
                                           align_extent(features.width, 4));
                }
                checkpoint("block" + std::to_string(block), features);
            }
        }

        for (block = 23; block <= 30; ++block) {
            current_block_ = block;
            features = split_window_block(features, block, shifts[(block - 23) % 4][0],
                                          shifts[(block - 23) % 4][1], &accumulator);
            checkpoint("block" + std::to_string(block), features);
        }
        skips[30] = features;
        auto pooled = pad_spatial(average_pool(accumulator), align_extent(features.height / 2, 4),
                                  align_extent(features.width / 2, 4));
        current_block_ = 30;
        current_layer_ = 4;
        features = matrix_multiply(pooled, model_weight(30, 4, "weight"), 1024, 5, true);
        checkpoint("pooled30", features);

        // Eight global transformer blocks operate on the coarsest features.
        for (block = 31; block <= 38; ++block) {
            demand(repeats[block - 31] <= 8, "Transformer repeat count exceeds eight");
            current_block_ = block;
            for (unsigned repetition = 0; repetition < repeats[block - 31]; ++repetition)
                features = transformer_block(features, block);
            checkpoint("block" + std::to_string(block), features);
        }

        // Decoder: restore resolution and combine the matching encoder skips.
        current_block_ = 39;
        features = upsample(features, skips[30], 39, true);
        checkpoint("block39", features);
        for (block = 40; block <= 47; ++block) {
            current_block_ = block;
            features = split_window_block(features, block, shifts[(block - 40) % 4][0],
                                          shifts[(block - 40) % 4][1]);
            checkpoint("block" + std::to_string(block), features);
        }
        const int level_sizes[] = {8, 6, 4, 4}, skip_blocks[] = {22, 14, 8, 4};
        block = 48;
        for (int level = 0; level < 4; ++level) {
            for (int index = 0; index < level_sizes[level]; ++index, ++block) {
                current_block_ = block;
                if (index == 0)
                    features = upsample(features, skips[skip_blocks[level]], block, false);
                int shift = (index + (level == 1 ? 2 : 0)) % 4;
                features =
                    shifted_window_block(features, block, shifts[shift][0], shifts[shift][1]);
                checkpoint("block" + std::to_string(block), features);
            }
            skips.erase(skip_blocks[level]);
        }

        // Blend the final decoder features with the stem skip, then project
        // the 32 feature channels to the four-channel output head.
        current_block_ = 70;
        current_layer_ = 0;
        auto blend_gains = upload_weight(
            "post_sincos",
            [&] {
                auto main_gain = model_.half(70, 0, "main_gain");
                auto skip_gain = model_.half(70, 0, "skip_gain");
                main_gain.insert(main_gain.end(), skip_gain.begin(), skip_gain.end());
                return main_gain;
            },
            false);
        features =
            shifted_window_block(features, 70, -1, -1, true, nullptr, &skips[0], &blend_gains);
        output = matrix_multiply(features,
                                 upload_weight("output_head",
                                               [&] {
                                                   auto values = model_.half(70, 0, "out_project");
                                                   values.resize(4 * 32);
                                                   return values;
                                               }),
                                 4);
        checkpoint("head", output);
        if (device_features_) {
            // These upload views live only before the first neural command and
            // share its existing scratch arena rather than adding resident buffers.
            frame_planes_[0] = allocate_tensor(height, width, 3, 4);
            frame_planes_[1] = allocate_tensor(height, width, 3, 4);
            frame_planes_[2] = allocate_tensor(height, width, 3, 2);
            // Validity bytes followed by an aligned optional RGBA32F map.
            frame_planes_[3] = allocate_tensor(height, width, 5, 4);
        }
        commit_scratch();
    }
