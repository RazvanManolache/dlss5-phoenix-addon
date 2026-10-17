import unittest
from probe_wow64_tls import appcompat_conclusions, conclusions, d3d12_conclusions, driver_bridge_conclusions, legacy_conclusions, model_parity_conclusions, portable_conclusions, symbol_rva, tls_success


class TlsProbeTests(unittest.TestCase):
    def parity_rows(self,backend='phoenix'):
        rows={}; frames={}
        for pid,bits,prefix in ((10,64,'native'),(11,32,'thread')):
            key=prefix+'_d3d12_'+backend
            rows[key]=dict(exit_code=0,process_pid=pid)
            sample=dict(changed_rgb_samples=1,evaluations=3,output_sha256='a'*64)
            frames[key]=dict(passed=True,gpu_copy_verified=True,process_pid=pid,process_bits=bits,
                pointer_bits=64,helper_pid=0,frames=[sample.copy() for _ in range(3 if backend=='phoenix' else 1)])
        return rows,frames

    def test_parity_requires_hashes_same_process_and_real_evaluations(self):
        for backend in ('phoenix','nvidia'):
            rows,frames=self.parity_rows(backend)
            result=model_parity_conclusions(rows,frames)[backend]
            self.assertTrue(result['same_process_inference'])
            self.assertTrue(result['bit_exact_against_x64'])
            self.assertFalse(result['game_presentation_validated'])
            key='thread_d3d12_'+backend
            frames[key]['frames'][0]['output_sha256']='b'*64
            result=model_parity_conclusions(rows,frames)[backend]
            self.assertTrue(result['same_process_inference'])
            self.assertFalse(result['bit_exact_against_x64'])
            for field,value in [('process_pid',12),('process_bits',64),('pointer_bits',32),('helper_pid',20)]:
                rows,frames=self.parity_rows(backend); frames[key][field]=value
                self.assertFalse(model_parity_conclusions(rows,frames)[backend]['same_process_inference'])

    def test_parity_refuses_incomplete_frames_crashes_and_missing_control(self):
        for backend in ('phoenix','nvidia'):
            key='thread_d3d12_'+backend
            for samples in ([],[dict(changed_rgb_samples=1,evaluations=1,output_sha256='a'*64)],
                            [dict(changed_rgb_samples=1,evaluations=3,output_sha256='invalid')]*3):
                rows,frames=self.parity_rows(backend); frames[key]['frames']=samples
                self.assertFalse(model_parity_conclusions(rows,frames)[backend]['same_process_inference'])
            rows,frames=self.parity_rows(backend); rows[key]['exit_code']=0xc0000005
            self.assertFalse(model_parity_conclusions(rows,frames)[backend]['same_process_inference'])
            rows,frames=self.parity_rows(backend); del rows['native_d3d12_'+backend]
            self.assertFalse(model_parity_conclusions(rows,frames)[backend]['same_process_inference'])

    def driver_rows(self):
        return dict(native_gpu=self.gpu(event_key='stage'), thread_gpu_native_driver=dict(
            process_pid=42, exit_code=2, timed_out=False, events=[
                dict(event='native_driver_bridge',pid=42,driver_pointer_bits=64,phoenix_runtime_loaded=False,bridged_escape_count=13),
                dict(event='native_thread_task',pid=42,caller_pointer_bits=32,completed=True),
                dict(event='gpu_worker_result',worker_stage=10,status=-1,module_created=0)]))

    def test_driver_calls_are_not_module_creation_or_inference(self):
        rows=self.driver_rows()
        result=driver_bridge_conclusions(rows)
        self.assertTrue(result['native_control'])
        self.assertTrue(result['same_process_driver_calls_observed'])
        self.assertFalse(result['module_creation_succeeded'])
        self.assertFalse(result['phoenix_inference_succeeded'])
        rows['thread_gpu_native_driver']['exit_code']=0
        rows['thread_gpu_native_driver']['events'][-1].update(worker_stage=12,status=0,module_created=1)
        self.assertTrue(driver_bridge_conclusions(rows)['module_creation_succeeded'])
        self.assertFalse(driver_bridge_conclusions(rows)['phoenix_inference_succeeded'])
        del rows['native_gpu']
        self.assertFalse(driver_bridge_conclusions(rows)['module_creation_succeeded'])

    def test_driver_thunk_requires_same_process_no_phoenix_and_observed_requests(self):
        for field,value in [('pid',43),('driver_pointer_bits',32),('phoenix_runtime_loaded',True),('bridged_escape_count',0)]:
            rows=self.driver_rows()
            rows['thread_gpu_native_driver']['events'][0][field]=value
            self.assertFalse(driver_bridge_conclusions(rows)['same_process_driver_calls_observed'])
        for field,value in [('pid',43),('caller_pointer_bits',64),('completed',False)]:
            rows=self.driver_rows()
            rows['thread_gpu_native_driver']['events'][1][field]=value
            self.assertFalse(driver_bridge_conclusions(rows)['same_process_driver_calls_observed'])
        for field,value in [('timed_out',True),('exit_code',0xc0000005),('process_pid',None)]:
            rows=self.driver_rows()
            rows['thread_gpu_native_driver'][field]=value
            self.assertFalse(driver_bridge_conclusions(rows)['same_process_driver_calls_observed'])

    def test_appcompat_fix_is_not_inference_and_needs_repeated_normal_exits(self):
        rows=dict(thread_frame_unseparated=dict(exit_code=0xc0000005,first_chance_faults=[dict(module='apphelp.dll',machine=0x14c)]),
            thread_frame=dict(exit_code=2),thread_frame_repeat=dict(exit_code=2))
        result=dict(error='no physical device (0)',frames=[],native_appcompat_view=True,native_appcompat_restored=True)
        frames={name:result.copy() for name in ('thread_frame','thread_frame_repeat')}
        conclusion=appcompat_conclusions(rows,frames)
        self.assertTrue(conclusion['x86_apphelp_fault_reproduced'])
        self.assertTrue(conclusion['separated_error_repeated'])
        self.assertTrue(conclusion['crash_guard_validated'])
        self.assertFalse(conclusion['inference_run'])
        frames['thread_frame_repeat']['native_appcompat_restored']=False
        self.assertFalse(appcompat_conclusions(rows,frames)['separated_error_repeated'])

    def test_clean_appcompat_control_does_not_validate_crash_guard(self):
        rows={name:dict(exit_code=2) for name in ('thread_frame_unseparated','thread_frame','thread_frame_repeat')}
        result=dict(error='no physical device (0)',frames=[],native_appcompat_view=True,native_appcompat_restored=True)
        frames={name:result.copy() for name in ('thread_frame','thread_frame_repeat')}
        conclusion=appcompat_conclusions(rows,frames)
        self.assertTrue(conclusion['separated_error_repeated'])
        self.assertFalse(conclusion['crash_guard_validated'])
        self.assertFalse(conclusion['inference_run'])
        frames['thread_frame_repeat']=result.copy(); rows['thread_frame_repeat']['exit_code']=0xc0000005
        self.assertFalse(appcompat_conclusions(rows,frames)['separated_error_repeated'])

    def test_portable_model_requires_processed_frames_and_control(self):
        rows={name:dict(exit_code=0) for name in ('native_frame','thread_frame')}
        result=dict(passed=True,frames=[dict(changed=12288,commands=1317) for _ in range(3)])
        self.assertTrue(portable_conclusions(rows,result,result)['same_process_x86_success'])
        for bad in ({},dict(passed=True,frames=[]),dict(passed=True,frames=[dict(changed=0,commands=1317)]*3),
                    dict(passed=True,frames=[dict(changed=12288,commands=0)]*3)):
            self.assertFalse(portable_conclusions(rows,result,bad)['same_process_x86_success'])
            self.assertFalse(portable_conclusions(rows,bad,result)['same_process_x86_success'])

    def test_portable_debugger_receipt_cannot_hide_normal_crash(self):
        rows=dict(native_frame=dict(exit_code=0),thread_frame=dict(exit_code=0xc0000005))
        result=dict(passed=True,frames=[dict(changed=12288,commands=1317)]*3)
        self.assertFalse(portable_conclusions(rows,result,result)['same_process_x86_success'])
        rows['thread_frame']=dict(exit_code=0,timed_out=True)
        self.assertFalse(portable_conclusions(rows,result,result)['same_process_x86_success'])

    def test_legacy_shader_is_never_neural_inference(self):
        self.assertIsNone(legacy_conclusions({},{}))
        rows={name:dict(exit_code=0) for name in ('native_d3d12_legacy','thread_d3d12_legacy')}
        result=dict(passed=True,gpu_copy_verified=True,shader_create_status=0,shader_created=True)
        frames={name:result.copy() for name in rows}
        classified=legacy_conclusions(rows,frames)
        self.assertTrue(classified['same_process_x86_success'])
        self.assertFalse(classified['inference_run'])
        self.assertEqual(classified['scope'],'shader_creation_only')
        frames['thread_d3d12_legacy']['shader_created']=False
        self.assertFalse(legacy_conclusions(rows,frames)['same_process_x86_success'])

    def test_model_success_needs_control_and_output(self):
        key='thread_d3d12_phoenix'
        rows={key:dict(exit_code=0)}
        frames={key:dict(passed=True,gpu_copy_verified=True,frames=[dict(changed_rgb_samples=1)])}
        self.assertFalse(d3d12_conclusions(rows,frames)['phoenix']['same_process_x86_success'])
        rows['native_d3d12_phoenix']=dict(exit_code=0)
        frames['native_d3d12_phoenix']=frames[key].copy()
        self.assertTrue(d3d12_conclusions(rows,frames)['phoenix']['same_process_x86_success'])
        frames[key]['frames']=[]
        self.assertFalse(d3d12_conclusions(rows,frames)['phoenix']['same_process_x86_success'])

    def test_nvidia_initialization_alone_is_not_inference(self):
        rows={key:dict(exit_code=0) for key in ('native_d3d12_nvidia','thread_d3d12_nvidia')}
        frames={key:dict(passed=True,gpu_copy_verified=True,initialize_status=1,frames=[]) for key in rows}
        self.assertFalse(d3d12_conclusions(rows,frames)['nvidia']['same_process_x86_success'])
        for frame in frames.values(): frame['frames']=[dict(changed_rgb_samples=1,evaluations=3)]
        self.assertTrue(d3d12_conclusions(rows,frames)['nvidia']['same_process_x86_success'])
        rows['thread_d3d12_nvidia']['exit_code']=0xc0000005
        self.assertFalse(d3d12_conclusions(rows,frames)['nvidia']['same_process_x86_success'])

    def test_unselected_backend_is_not_marked_failed(self):
        self.assertEqual(d3d12_conclusions({},{}),dict(cuda=None,phoenix=None,nvidia=None))

    def good(self):
        return dict(exit_code=0, timed_out=False, events=[dict(event='tls_result', stage=5,
            template_ok=True, zero_fill_ok=True)])

    def test_template_and_zero_fill_are_both_required(self):
        self.assertTrue(tls_success(self.good()))
        for key in ('template_ok', 'zero_fill_ok'):
            row = self.good()
            row['events'][0][key] = False
            self.assertFalse(tls_success(row))

    def test_returning_success_without_storage_does_not_count(self):
        row = self.good()
        row['events'][0]['stage'] = 4
        self.assertFalse(tls_success(row))
        self.assertFalse(tls_success(dict(exit_code=0, events=[])))

    def test_crash_and_timeout_are_failures(self):
        row = self.good()
        row['exit_code'] = 0xc0000005
        self.assertFalse(tls_success(row))
        row = self.good()
        row['timed_out'] = True
        self.assertFalse(tls_success(row))

    def test_cpu_tls_success_does_not_claim_phoenix_or_gpu(self):
        result = conclusions({key: self.good() for key in
            ('native_tls', 'cross_tls', 'cross_repair', 'thread_tls', 'thread_repair')})
        self.assertTrue(result['native_control'])
        self.assertTrue(result['cross_repaired_tls'])
        for key in ('phoenix_route_implemented', 'gpu_device_created', 'inference_run'):
            self.assertFalse(result[key])

    def test_handler_address_must_be_a_symbol_derived_rva(self):
        self.assertEqual(symbol_rva('PHX_TLS_RVA=00000000`00012345'), 0x12345)
        for text in ('PHX_TLS_RVA=00000000', 'PHX_TLS_RVA=00007fff`00012345', 'unknown symbol'):
            self.assertIsNone(symbol_rva(text))

    def test_loading_library_and_defaults_are_not_error_or_inference_success(self):
        rows = {'initialized_native': dict(exit_code=0xe06d7363, events=[
            dict(event='phoenix_load_result', loaded=True), dict(event='phoenix_defaults', matched=True)])}
        result = conclusions(rows)
        self.assertTrue(result['phoenix_library_loaded_before_fault'])
        self.assertTrue(result['phoenix_defaults_matched_before_fault'])
        for key in ('phoenix_x64_cpu_api', 'phoenix_route_implemented', 'inference_run'):
            self.assertFalse(result[key])

    def gpu(self, stage=12, status=0, event_key='event'):
        return dict(exit_code=0, timed_out=False, events=[dict(
            **{event_key: 'worker_result'}, worker_stage=stage, status=status, module_created=1)])

    def test_gpu_success_requires_positive_control_and_complete_worker(self):
        cross = self.gpu()
        self.assertFalse(conclusions(dict(thread_gpu=cross))['native_thread_gpu_module'])
        rows = dict(native_gpu=self.gpu(event_key='stage'), thread_gpu=cross)
        self.assertTrue(conclusions(rows)['native_thread_gpu_module'])
        self.assertTrue(conclusions(rows)['gpu_device_created'])
        for key in ('phoenix_route_implemented', 'inference_run'):
            self.assertFalse(conclusions(rows)[key])
        for stage, status in ((4, -2005270524), (10, 0), (12, -1)):
            rows['thread_gpu'] = self.gpu(stage, status)
            self.assertFalse(conclusions(rows)['native_thread_gpu_module'])

    def test_timed_out_or_crashed_gpu_cannot_pass(self):
        for key, value in (('timed_out', True), ('exit_code', 0xe06d7363)):
            rows = dict(native_gpu=self.gpu(), thread_gpu=self.gpu())
            rows['thread_gpu'][key] = value
            self.assertFalse(conclusions(rows)['native_thread_gpu_module'])

    def test_created_device_does_not_imply_created_kernel(self):
        probe = self.gpu(stage=10, status=-1)
        probe['exit_code'] = 2
        probe['events'][0]['module_created'] = 0
        result = conclusions(dict(native_gpu=self.gpu(), thread_gpu_native_umd=probe))
        self.assertTrue(result['gpu_device_created'])
        self.assertTrue(result['gpu_queue_created'])
        self.assertFalse(result['redirected_native_thread_gpu_module'])
        self.assertFalse(result['inference_run'])

    def test_native_thread_error_requires_valid_stack_and_native_control(self):
        row = dict(exit_code=0, events=[dict(event='native_thread_task', completed=True,
            defaults_ok=True, status=-1, message='Null native session', stack_address=15,
            stack_limit=10, stack_base=20)])
        rows = dict(thread_native=row)
        self.assertFalse(conclusions(rows)['phoenix_x64_cpu_api'])
        rows['native_phoenix_api'] = dict(exit_code=0, events=[dict(
            event='native_phoenix_error_control', status=-1, message='Null native session')])
        self.assertTrue(conclusions(rows)['phoenix_x64_cpu_api'])
        row['events'][0]['stack_address'] = 30
        self.assertFalse(conclusions(rows)['phoenix_x64_cpu_api'])


if __name__ == '__main__':
    unittest.main()
