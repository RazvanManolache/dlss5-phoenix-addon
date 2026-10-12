"""Auditable graph adaptation to DLSS5 Phoenix arithmetic, applied only to new builds."""

from source_patches import code_count, replace_code


def once(text, old, new):
    if code_count(text, old) != 1:
        raise RuntimeError("DLSS5 Phoenix math anchor: " + old[:100])
    return replace_code(text, old, new)


def patch(gpu, out, own):
    path = gpu / "publish.glsl"
    source_text = path.read_text()
    # Replace only the publication converter; keep imported files untouched.
    begin = source_text.index("float e4m3(float x) {")
    end = source_text.index("\n}", begin) + 2
    source_text = once(source_text, source_text[begin:end], (own / "fp8_round.glsl").read_text())
    source_text = once(
        source_text,
        "float publish(uint epilogue, float value) {",
        """float publish(uint epilogue, float value) {
    if(epilogue==6u) {
        float t=clamp(value,-4.0,4.0);
        return e4m3(value*fma(t,fma(abs(t),-0.055908203125,0.447265625),0.89453125));
    }
    if(epilogue==7u) {
        float t=clamp(value,-4.0,4.0);
        precise float a=-0.055908203125*abs(t)*t;
        precise float b=0.447265625*t;
        precise float v=value*((a+b)+0.89453125);
        return e4m3(half_round(v));
    }""",
    )
    path.write_text(source_text)
    path = gpu / "resident.comp"
    source_text = path.read_text()
    # Some AMD drivers flush an FP32 subnormal when multiplying by one.
    # Identity scaling must preserve its sign for the integer quantizers below.
    source_text = once(
        source_text,
        "float value = load_a(index) * pc.p0;",
        "float value = pc.p0 == 1.0 ? load_a(index) : load_a(index) * pc.p0;",
    )
    source_text = once(
        source_text,
        "    uint kind = operation_flags() & 0xFFu;",
        """    uint kind = operation_flags() & 0xFFu;
    if(kind==23u) {
        uint code=(WeightWords(uint64_t(pc.a)).v[index>>2u]>>((index&3u)*8u))&255u;
        uint e=(code>>3u)&15u,m=code&7u,sign=(code&128u)<<24u;
        uint bits=e!=0u?((e+120u)<<23u)|(m<<20u):floatBitsToUint(float(m)*0.001953125);
        HalfBuf(uint64_t(pc.c)).v[index]=float16_t(uintBitsToFloat(sign|bits));return;
    }""",
    )
    source_text = once(
        source_text,
        "void main() {",
        "layout(buffer_reference,scalar,buffer_reference_align=4) buffer WeightWords {uint v[];};\nvoid main() {",
    )
    source_text = once(
        source_text,
        "    if(kind==23u) {",
        (own / "streamed_window.glsl").read_text() + "\n    if(kind==23u) {",
    )
    # Window token order is tile-row/tile-column/pixel-row/pixel-column.
    source_text = once(
        source_text,
        "+ (py % size) * size + px % size) * channels + c;",
        "+ ((py%8u)/4u)*32u+((px%8u)/4u)*16u+(py%4u)*4u+px%4u) * channels + c;",
    )
    source_text = once(
        source_text,
        "+ token / size) - int(pad_top);",
        "+ (token/32u)*4u+(token%16u)/4u) - int(pad_top);",
    )
    source_text = once(
        source_text,
        "+ token % size) - int(pad_left);",
        "+ ((token/16u)%2u)*4u+token%4u) - int(pad_left);",
    )
    source_text = once(
        source_text,
        "uint token = (py % size) * size + (px % size);",
        "uint token = ((py%8u)/4u)*32u+((px%8u)/4u)*16u+(py%4u)*4u+px%4u;",
    )
    source_text = once(
        source_text,
        """precise float total = load_a(base) + load_a(base + width * channels);
        total += load_a(base + channels);
        total += load_a(base + (width + 1u) * channels);""",
        """precise float top = half_round(load_a(base)+load_a(base+channels));
        precise float bottom = half_round(load_a(base+width*channels)+load_a(base+(width+1u)*channels));
        precise float total=half_round(top+bottom);""",
    )
    source_text = once(
        source_text,
        "store(index, load_a(at) + load_b(index) * pc.d.v[index % pc.n]);",
        "precise float product=load_b(index)*pc.d.v[index%pc.n];\n        precise float sum=load_a(at)+product;\n        store(index,sum);",
    )
    source_text = once(
        source_text,
        """        float merged = fma(load_b(index), pc.d.v[channels + c], scaled);
        pc.c.v[index] = merged;
        HalfBuf(pc.second).v[index] = float16_t(e4m3(merged));""",
        """        // Match the standalone scale -> half -> ordered residual -> half.
        float merged = half_round(fma(load_b(index), pc.d.v[channels+c], half_round(scaled)));
        pc.c.v[index] = merged;
        HalfBuf(pc.second).v[index] = float16_t(e4m3(merged));""",
    )
    source_text = once(
        source_text,
        """        precise float total = value[0] + value[1];
        total += value[2];
        total += value[3];""",
        """        precise float top=half_round(value[0]+value[2]);
        precise float bottom=half_round(value[1]+value[3]);
        precise float total=half_round(top+bottom);""",
    )
    source_text = once(
        source_text,
        "HalfBuf(pc.second).v[taps[t]] = float16_t(e4m3(value[t]));",
        "HalfBuf(pc.second).v[taps[t]] = float16_t(e4m3(half_round(value[t])));",
    )
    source_text = once(
        source_text,
        """precise float scaled = load_b(index) * pc.d.v[c];
        precise float merged = load_a(((y / 2u) * source + (x / 2u)) * channels + c) + scaled;""",
        """float projected=half_round(load_a(((y/2u)*source+(x/2u))*channels+c));
        float merged=channels==512u ? half_round(projected+half_round(load_b(index)*pc.d.v[c]))
                                   : half_round(fma(load_b(index),pc.d.v[c],projected));""",
    )
    path.write_text(source_text)
    path = gpu / "attention.comp"
    source_text = path.read_text()
    # These helpers need stage, buffers and bias_offset declared before them.
    source_text = once(
        source_text,
        "void softmax(uint row, uint local) {",
        (own / "phoenix_swin.glsl").read_text()
        + "\nvoid softmax(uint row, uint local) {\n    if(pc.n==64u) {if(pc.batch!=16u) phoenix_swin_softmax(row,local); else phoenix_split_softmax(row,local);return;}",
    )
    source_text = once(
        source_text,
        "if (row < pc.m) cosine_publish(row, local, pc.k != 0u, pc.d);",
        "if(row<pc.m) { if(pc.batch!=16u && (operation_flags()&0x80000u)==0u) phoenix_swin_cosine(row,local,pc.k!=0u,pc.d); else phoenix_half_cosine(row,local,pc.k!=0u,pc.d,(operation_flags()&0x80000u)!=0u); }",
    )
    path.write_text(source_text)
    path = out / "nr_resident.py"
    source_text = path.read_text()
    source_text = once(
        source_text,
        '            self.expand = take("weight1")',
        '            self.hidden_width = weights[f"{prefix}.weight1"].shape[1]\n            self.expand = take("weight1")',
    )
    source_text = once(
        source_text,
        "weights.expand.nbytes // 2 // channels",
        'weights.hidden_width if not getattr(weights, "split", False) else 0',
    )
    # Direct FP32 Q/K normalization avoids the intermediate half split on Swin.
    source_text = once(
        source_text, "    if runtime.fuse_qk:\n", "    if runtime.fuse_qk or not vit:\n"
    )
    # V always has an FP16 rounding before the E4M3 store.
    source_text = replace_code(source_text, "(5 if vit else xmxres.EPI_E4M3)", "5")
    section_begin = source_text.index("def record_feed_forward(")
    section_end = source_text.index("def record_split_feed_forward(", section_begin)
    section_text = replace_code(
        replace_code(source_text[section_begin:section_end], "xmxres.EPI_GATE_E4M3", "6"),
        "epilogue=xmxres.EPI_E4M3",
        "epilogue=5",
    )
    source_text = source_text[:section_begin] + section_text + source_text[section_end:]
    section_begin = source_text.index("def record_split_feed_forward(")
    section_end = source_text.index("def can_fuse_window_block(", section_begin)
    section_text = replace_code(
        replace_code(
            source_text[section_begin:section_end], "epilogue=xmxres.EPI_E4M3", "epilogue=5"
        ),
        "epilogue=xmxres.EPI_GATE)",
        "epilogue=xmxres.EPI_GATE_E4M3)",
    )
    source_text = source_text[:section_begin] + section_text + source_text[section_end:]
    section_begin = source_text.index("def record_global_block(")
    section_end = source_text.index("def run_global_block(", section_begin)
    section_text = replace_code(
        replace_code(
            source_text[section_begin:section_end], "epilogue=xmxres.EPI_GATE_E4M3", "epilogue=7"
        ),
        "epilogue=xmxres.EPI_E4M3",
        "epilogue=5",
    )
    # The public global block leaves a raw output; the enclosing graph publishes it.
    source_text = source_text[:section_begin] + section_text + source_text[section_end:]
    anchor = "epilogue=xmxres.EPI_E4M3 if publish_image else 0)"
    if code_count(source_text, anchor) != 2:
        raise RuntimeError("Partition publish anchors")
    source_text = replace_code(source_text, anchor, "epilogue=5 if publish_image else 0)")
    source_text = once(
        source_text,
        "runtime.merge_heads(s.context, s.merged16, windows, tokens, channels, heads,\n                            epilogue=xmxres.EPI_E4M3, narrow=True)",
        "runtime.merge_heads(s.context, s.merged16, windows, tokens, channels, heads,\n                            epilogue=5, narrow=True)",
    )
    source_text = once(
        source_text,
        "epilogue=publish, narrow=target_half, b_half=ffn_half)",
        "epilogue=5 if publish==xmxres.EPI_E4M3 else publish, narrow=target_half, b_half=ffn_half)",
    )
    source_text = once(
        source_text,
        "transition.out_channels, channels, epilogue=xmxres.EPI_E4M3,",
        "transition.out_channels, channels, epilogue=5,",
    )
    source_text = once(
        source_text,
        "    if runtime.fuse_transition:\n",
        "    if True:  # DLSS5 Phoenix decoder blend has explicit FP16 boundaries.\n",
    )
    # Bound every window block's intermediates to at most 256 windows. The
    # original full-resolution extent remains available for gather/scatter.
    source_text = once(
        source_text,
        "        pixels = height * width",
        """        pixels = height * width
        if getattr(runtime,"phoenix_streamed_window",False):
            self.image_height,self.image_width=height,width
            self.windows=min(self.windows,256)
            self.batch=self.windows*heads
            pixels=self.windows*64
            self.height,self.width=1,pixels""",
    )
    source_text = once(
        source_text,
        "        self.hidden_width = hidden",
        """        self.hidden_width = hidden
        if getattr(runtime,"phoenix_streamed_window",False) and channels==32:
            self.stream_features16=make("stream.features",pixels*16,np.float16)
            self.stream_head4=make("stream.head",pixels*4)""",
    )
    start = source_text.index("def record_block(")
    prefix = source_text[:start]
    body = source_text[start:]
    body = once(
        body,
        "    source = source or s.value",
        """    if getattr(runtime,"phoenix_streamed_window",False):
        from phoenix_portable.streamed_window import record_streamed_block
        return record_streamed_block(runtime,w,s,source,target,publish,source_half,target_half,
                                     stem=stem,merge=merge,pool=pool,head=head)
    source = source or s.value""",
    )
    source_text = prefix + body
    start = source_text.index("class BlockScratch:")
    end = source_text.index("def _ffn_groups(", start)
    body = source_text[start:end]
    body = once(
        body,
        'self.ffn = make("ffn", pixels * channels)',
        'self.ffn = make("ffn", pixels * channels, np.float16 if getattr(runtime,"phoenix_streamed_window",False) and (weights.branched or getattr(weights,"split",False)) else np.float32)',
    )
    source_text = source_text[:start] + body + source_text[end:]
    # Global attention still sees all keys for each query. Only query rows are
    # batched, so exponential/denominator reduction order remains unchanged.
    # Future query rows remain live after each batch publishes probabilities.
    # They cannot share the query_probability arena used by the original
    # whole-matrix schedule, where all queries were consumed first.
    source_text = once(
        source_text,
        'make(name, padded * channels, np.float16) for name in ("q16", "k16", "v16")',
        'make("global.query" if name=="q16" and getattr(runtime,"phoenix_streamed_window",False) else name, padded * channels, np.float16) for name in ("q16", "k16", "v16")',
    )
    source_text = once(
        source_text,
        '        self.scores = make("scores", heads * padded * padded)',
        '        query_rows=min(padded,64) if getattr(runtime,"phoenix_streamed_window",False) else padded\n        self.scores = make("scores", heads * query_rows * padded)',
    )
    source_text = once(
        source_text,
        '        self.probs16 = make("probs16", heads * padded * padded, np.float16)',
        '        self.probs16 = make("probs16", heads * query_rows * padded, np.float16)',
    )
    source_text = once(
        source_text,
        '        self.reciprocal = make("reciprocal", heads * padded)',
        '        self.reciprocal = make("reciprocal", heads * query_rows)',
    )
    source_text = once(
        source_text,
        '        self.attention = make("attention", padded * channels)',
        '        self.global_merged16=make("global.merged",padded*channels,np.float16)\n        self.attention = make("attention", padded * channels)',
    )
    source_text = once(
        source_text,
        "    if runtime.fuse_global_attention:",
        """    if getattr(runtime,"phoenix_streamed_window",False):
        from phoenix_portable.streamed_window import record_streamed_global_attention
        merged=record_streamed_global_attention(runtime,w,s,key)
    elif runtime.fuse_global_attention:""",
    )
    # Transition scratch follows actual producer dimensions, rather than a
    # rounded worst-case extent shared by unrelated inputs and outputs.
    source_text = once(
        source_text,
        '        self.padded = make("transition.padded", elements)',
        """        self._make=make;self._exact={}
        if getattr(runtime,"phoenix_streamed_window",False):
            self.padded=self.pooled16=self.projected=self.projected16=self.upsampled=self.scaled=None
            return
        self.padded = make("transition.padded", elements)""",
    )
    start = source_text.index("class TransitionScratch:")
    body = source_text[start:]
    body = once(
        body,
        "    def free(self):",
        """    def exact(self,name,count,dtype=np.float32):
        key=(name,count,np.dtype(dtype).str)
        if key not in self._exact:self._exact[key]=self._make("transition."+name,count,dtype)
        return self._exact[key]

    def free(self):""",
    )
    source_text = source_text[:start] + body
    source_text = once(
        source_text,
        "        runtime.pad_end(source, scratch.padded, height, width, padded_height,",
        '        if getattr(runtime,"phoenix_streamed_window",False):scratch.padded=scratch.exact("padded",padded_height*padded_width*channels,np.float16 if source_half else np.float32)\n        runtime.pad_end(source, scratch.padded, height, width, padded_height,',
    )
    source_text = once(
        source_text,
        "    runtime.pool2(source, scratch.pooled16, height, width, channels,",
        '    if getattr(runtime,"phoenix_streamed_window",False):scratch.pooled16=scratch.exact("pooled16",(height//2)*(width//2)*channels,np.float16)\n    runtime.pool2(source, scratch.pooled16, height, width, channels,',
    )
    source_text = once(
        source_text,
        "    projected16 = source if source_half else scratch.projected16",
        """    if getattr(runtime,"phoenix_streamed_window",False):
        scratch.projected=scratch.exact("projected",source_pixels*out_channels)
        if not source_half:scratch.projected16=scratch.exact("projected16",source_pixels*channels,np.float16)
    projected16 = source if source_half else scratch.projected16""",
    )
    source_text = once(
        source_text,
        "                          target_half=False, published=None):",
        "                          target_half=False, published=None, raw_output=False):",
    )
    source_text = once(
        source_text,
        "    epilogue = xmxres.EPI_E4M3 if published is None else 0",
        "    epilogue = xmxres.EPI_E4M3 if published is None and not raw_output else 0",
    )
    path.write_text(source_text)
    path = out / "nr_frame_resident.py"
    source_text = path.read_text()
    # These roles have disjoint lifetimes in the recorded graph: stem is dead
    # before the decoder merge, and every raw transition is consumed before
    # the next one. Keep their allocations separate from persistent skips.
    source_text = once(
        source_text,
        "        existing = self._buffers.get(name)",
        """        aliases={"stem":"frame.pre_post", "merged":"frame.pre_post",
                 "block0":"frame.transient", "unpublished":"frame.transient",
                 "raw30":"frame.transient", "out16":"frame.transient"}
        name=aliases.get(name,name)
        existing = self._buffers.get(name)""",
    )
    source_text = once(
        source_text,
        "        name=aliases.get(name,name)",
        """        if getattr(self.rt,"phoenix_streamed_window",False):
            # Each decoder merge is the final consumer of its encoder skip.
            # Its elementwise read precedes the same-index write in that pass.
            aliases.update({f"d{i}":f"l{i}" for i in range(1,6)})
        name=aliases.get(name,name)""",
    )
    source_text = once(
        source_text,
        "rt.gemm(source16, self.adapter, stem, pixels, 32, 16)",
        "rt.gemm(source16, self.adapter, stem, pixels, 32, 16, epilogue=4)",
    )
    # Skip conversion has double rounding; pool still consumes the raw accumulator.
    source_text = once(
        source_text,
        "rt.e4m3_half(raw, full_skip, pixels * 32)",
        "rt.unary(2, raw, full_skip, pixels * 32, epilogue=5, narrow=True)",
    )
    source_text = once(
        source_text,
        "rt.e4m3_half(unpublished, value, h * w * channels)",
        "rt.unary(2, unpublished, value, h * w * channels, epilogue=5, narrow=True)",
    )
    source_text = once(
        source_text,
        "rt.e4m3(scratch.out, scratch.out, scratch.padded * gchannels)",
        "rt.unary(2, scratch.out, scratch.out, scratch.padded * gchannels, epilogue=5)",
    )
    source_text = once(
        source_text,
        "rt.e4m3_half(raw30, value, h * w * channels)",
        "rt.unary(2, raw30, raw30, h * w * channels)\n                rt.e4m3_half(raw30, value, h * w * channels)",
    )
    source_text = once(
        source_text,
        "rt.scale_channel(upsampled, self.merge_sin, merged, pixels * 32, 32)",
        "rt.scale_channel(upsampled, self.merge_sin, merged, pixels * 32, 32)\n            rt.unary(2, merged, merged, pixels*32)",
    )
    source_text = once(
        source_text,
        "rt.residual(merged, full_skip, self.merge_cos, merged, pixels * 32, 32,\n                        b_half=True)",
        "rt.residual(merged, full_skip, self.merge_cos, merged, pixels * 32, 32,\n                        b_half=True, epilogue=4)",
    )
    source_text = once(
        source_text,
        """            upsampled = self.buffer("upsampled", pixels * 32)
            rt.upsample2(value, upsampled, w, height, width, 32, a_half=True)
            rt.scale_channel(upsampled, self.merge_sin, merged, pixels * 32, 32)
            rt.unary(2, merged, merged, pixels*32)
            rt.residual(merged, full_skip, self.merge_cos, merged, pixels * 32, 32,
                        b_half=True, epilogue=4)""",
        """            rt.upsample_merge(value, full_skip, self.merge_sincos, merged,
                              scratch70.value16, height, width, w, 32)""",
    )
    source_text = once(
        source_text,
        "source16=scratch70.value16 if rt.fuse_glue and not merge else None,",
        "source16=scratch70.value16 if not merge else None,",
    )
    source_text = once(
        source_text,
        "elif rt.fuse_glue and height % 2 == 0 and width % 2 == 0:",
        "elif height % 2 == 0 and width % 2 == 0:",
    )
    source_text = once(
        source_text,
        "made_stem = rt.fuse_glue and capture is None and R.can_make_stem(rt, block0, scratch0)",
        'made_stem = (getattr(rt,"phoenix_streamed_window",False) and capture is None) or (rt.fuse_glue and capture is None and R.can_make_stem(rt, block0, scratch0))',
    )
    source_text = once(
        source_text,
        "pooled_here = rt.fuse_glue and capture is None and R.can_pool_output(rt, block0, scratch0)",
        'pooled_here = (getattr(rt,"phoenix_streamed_window",False) and capture is None) or (rt.fuse_glue and capture is None and R.can_pool_output(rt, block0, scratch0))',
    )
    source_text = once(
        source_text,
        "fused_head = (rt.fuse_head and capture is None",
        'fused_head = (getattr(rt,"phoenix_streamed_window",False) and capture is None) or (rt.fuse_head and capture is None',
    )
    source_text = once(
        source_text,
        "if rt.fuse_glue and R.can_merge_input(rt, block70, scratch70):",
        'if (getattr(rt,"phoenix_streamed_window",False) and capture is None) or (rt.fuse_glue and R.can_merge_input(rt, block70, scratch70)):',
    )
    source_text = once(
        source_text,
        "published=None if raw is None else target)",
        'published=None if raw is None or getattr(rt,"phoenix_streamed_window",False) else target,\n                raw_output=raw is not None and getattr(rt,"phoenix_streamed_window",False))',
    )
    path.write_text(source_text)
