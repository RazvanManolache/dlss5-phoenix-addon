// Real legacy DirectInput, hidden test-owned window, no desktop input or GPU.
#include "../../src/integration/input/phoenix_input_addon.cpp"
#include <iostream>
static unsigned test_chars=0,test_keys=0;
static HWND test_foreground=nullptr;
static unsigned cursor_writes=0,clip_writes=0;
static HWND WINAPI test_GetForegroundWindow(){return test_foreground;}
static BOOL WINAPI test_GetCursorPos(LPPOINT p){*p={900,901};return TRUE;}
static BOOL WINAPI test_SetCursorPos(int,int){++cursor_writes;return TRUE;}
static BOOL WINAPI test_ClipCursor(const RECT*){++clip_writes;return TRUE;}
static LRESULT CALLBACK test_wndproc(HWND hwnd,UINT message,WPARAM wp,LPARAM lp) {
    if(message==WM_CHAR)++test_chars;
    if(message==WM_KEYDOWN)++test_keys;
    return DefWindowProcW(hwnd,message,wp,lp);
}
int main() {
    auto legacy=LoadLibraryW(L"dinput.dll");
    if(!legacy)return 2;
    using Create=HRESULT(WINAPI*)(HINSTANCE,DWORD,REFIID,LPVOID*,LPUNKNOWN);
    auto create=reinterpret_cast<Create>(GetProcAddress(legacy,"DirectInputCreateEx"));
    if(!create)return 3;
    auto hwnd=CreateWindowExW(0,L"STATIC",L"Phoenix legacy input test",WS_OVERLAPPED,0,0,32,32,nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);
    if(!hwnd)return 4;
    g_hwnd=hwnd;g_focus=true;
    SetWindowLongPtrW(hwnd,GWLP_WNDPROC,reinterpret_cast<LONG_PTR>(&test_wndproc));
    for(int key:{VK_RETURN,VK_ESCAPE}) {
        test_chars=test_keys=0;set_key(key,true);set_key(key,false);
        MSG message{};unsigned pumped=0;
        while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE)&&++pumped<64) { TranslateMessage(&message);DispatchMessageW(&message); }
        if(test_keys!=1||test_chars!=1) { std::cerr<<"Duplicate/missing characters for key "<<key<<": "<<test_chars<<" from "<<test_keys<<" keys\n";return 21; }
    }
    // Simulate an owned foreground window without activating it or moving the
    // desktop cursor. A scripted pointer must remain authoritative in this case.
    test_foreground=hwnd;o_GetForegroundWindow=test_GetForegroundWindow;
    o_GetCursorPos=test_GetCursorPos;o_SetCursorPos=test_SetCursorPos;o_ClipCursor=test_ClipCursor;
    POINT actual{},expected{7,9};ClientToScreen(hwnd,&expected);
    if(!h_GetCursorPos(&actual)||actual.x!=900||actual.y!=901)return 28;
    move_mouse(0,0,true,7,9);
    if(!h_GetCursorPos(&actual)||actual.x!=expected.x||actual.y!=expected.y){std::cerr<<"Foreground scripted cursor ignored\n";return 22;}
    POINT warp{13,15};ClientToScreen(hwnd,&warp);
    if(!h_SetCursorPos(warp.x,warp.y)||cursor_writes||g_cursor.x!=13||g_cursor.y!=15)return 23;
    RECT clip{0,0,16,16};if(!h_ClipCursor(&clip)||clip_writes)return 24;
    if(!h_ClipCursor(nullptr)||clip_writes!=1)return 25;
    release_all();if(!h_GetCursorPos(&actual)||actual.x!=900||actual.y!=901)return 29;
    g_focus=false;
    if(!h_GetCursorPos(&actual)||actual.x!=900||actual.y!=901)return 26;
    if(!h_SetCursorPos(0,0)||cursor_writes!=1||!h_ClipCursor(&clip)||clip_writes!=2)return 27;
    o_GetForegroundWindow=nullptr;o_GetCursorPos=nullptr;o_SetCursorPos=nullptr;o_ClipCursor=nullptr;g_focus=true;
    if(MH_Initialize()!=MH_OK)return 5;
    install_dinput();
    unsigned checks=0;
    for(bool modern:{false,true}) for(bool wide:{false,true}) {
        IUnknown* root=nullptr;
        auto factory=modern?reinterpret_cast<Create>(GetProcAddress(GetModuleHandleW(L"dinput8.dll"),"DirectInput8Create")):create;
        if(!factory)return 13;
        auto hr=factory(GetModuleHandleW(nullptr),modern?0x800:0x700,modern?(wide?IID_IDirectInput8W:IID_IDirectInput8A):(wide?IID_IDirectInput7W:IID_IDirectInput7A),reinterpret_cast<void**>(&root),nullptr);
        if(FAILED(hr)||!root)return 6;
        void* device=nullptr;
        hr=modern?(wide?static_cast<IDirectInput8W*>(static_cast<void*>(root))->CreateDevice(GUID_SysKeyboard,reinterpret_cast<LPDIRECTINPUTDEVICE8W*>(&device),nullptr)
                       :static_cast<IDirectInput8A*>(static_cast<void*>(root))->CreateDevice(GUID_SysKeyboard,reinterpret_cast<LPDIRECTINPUTDEVICE8A*>(&device),nullptr))
                 :wide?static_cast<IDirectInput7W*>(static_cast<void*>(root))->CreateDevice(GUID_SysKeyboard,reinterpret_cast<LPDIRECTINPUTDEVICEW*>(&device),nullptr)
               :static_cast<IDirectInput7A*>(static_cast<void*>(root))->CreateDevice(GUID_SysKeyboard,reinterpret_cast<LPDIRECTINPUTDEVICEA*>(&device),nullptr);
        if(FAILED(hr)||!device)return 7;
        auto* keys=static_cast<IDirectInputDeviceA*>(device);
        if(FAILED(keys->SetDataFormat(&c_dfDIKeyboard)))return 8;
        if(FAILED(keys->SetCooperativeLevel(hwnd,DISCL_BACKGROUND|DISCL_NONEXCLUSIVE)))return 9;
        DIPROPDWORD buffer{};buffer.diph={sizeof(buffer),sizeof(buffer.diph),0,DIPH_DEVICE};buffer.dwData=16;
        if(FAILED(keys->SetProperty(DIPROP_BUFFERSIZE,&buffer.diph)))return 10;
        keys->Acquire();
        g_vk[VK_F8]=0x80;
        unsigned char state[256]{};
        hr=keys->GetDeviceState(sizeof(state),state);
        if(FAILED(hr)||state[DIK_F8]!=0x80){std::cerr<<"legacy keyboard state missing, wide="<<wide<<" hr="<<std::hex<<hr<<std::endl;return 11;}
        ++checks;g_vk[VK_F8]=0;
        // Games may create/format devices before the first swapchain installs hooks.
        g_di_kind.clear();
        {std::lock_guard lock(g_lock);g_di_keyboard_events.clear();g_di_keyboard_events.push_back({DIK_F8,0x80});}
        DIDEVICEOBJECTDATA events[16]{};DWORD count=16;
        count=INFINITE;
        hr=keys->GetDeviceData(sizeof(events[0]),nullptr,&count,DIGDD_PEEK);
        if(FAILED(hr)||count!=1||g_di_keyboard_events.size()!=1)return 23;
        ++checks;
        count=0;hr=keys->GetDeviceData(sizeof(events[0]),nullptr,&count,0);
        if(FAILED(hr)||count!=0||g_di_keyboard_events.size()!=1)return 24;
        ++checks;
        for(int peek=0;peek<2;++peek) {
            count=16;hr=keys->GetDeviceData(sizeof(events[0]),events,&count,DIGDD_PEEK);
            bool visible=false;for(DWORD i=0;i<count;++i)visible|=events[i].dwOfs==DIK_F8&&events[i].dwData==0x80;
            if(FAILED(hr)||!visible||g_di_keyboard_events.size()!=1)return 14;
            ++checks;
        }
        count=16;
        hr=keys->GetDeviceData(sizeof(events[0]),events,&count,0);
        bool found=false;for(DWORD i=0;i<count;++i)found|=events[i].dwOfs==DIK_F8&&events[i].dwData==0x80;
        if(FAILED(hr)||!found)return 12;
        ++checks;
        {std::lock_guard lock(g_lock);g_di_keyboard_events.push_back({DIK_F8,0});g_di_keyboard_events.push_back({DIK_F8,0x80});}
        count=1;hr=keys->GetDeviceData(sizeof(events[0]),nullptr,&count,0);
        if(FAILED(hr)||count!=1||g_di_keyboard_events.size()!=1)return 25;
        ++checks;count=INFINITE;
        hr=keys->GetDeviceData(sizeof(events[0]),nullptr,&count,0);
        if(FAILED(hr)||count!=1||!g_di_keyboard_events.empty())return 26;
        ++checks;keys->Unacquire();keys->Release();
        device=nullptr;
        hr=modern?(wide?static_cast<IDirectInput8W*>(static_cast<void*>(root))->CreateDevice(GUID_SysMouse,reinterpret_cast<LPDIRECTINPUTDEVICE8W*>(&device),nullptr)
                       :static_cast<IDirectInput8A*>(static_cast<void*>(root))->CreateDevice(GUID_SysMouse,reinterpret_cast<LPDIRECTINPUTDEVICE8A*>(&device),nullptr))
                 :wide?static_cast<IDirectInput7W*>(static_cast<void*>(root))->CreateDevice(GUID_SysMouse,reinterpret_cast<LPDIRECTINPUTDEVICEW*>(&device),nullptr)
                      :static_cast<IDirectInput7A*>(static_cast<void*>(root))->CreateDevice(GUID_SysMouse,reinterpret_cast<LPDIRECTINPUTDEVICEA*>(&device),nullptr);
        if(FAILED(hr)||!device)return 15;
        auto* mouse=static_cast<IDirectInputDeviceA*>(device);
        if(FAILED(mouse->SetDataFormat(&c_dfDIMouse2))||FAILED(mouse->SetCooperativeLevel(hwnd,DISCL_BACKGROUND|DISCL_NONEXCLUSIVE)))return 16;
        if(FAILED(mouse->SetProperty(DIPROP_BUFFERSIZE,&buffer.diph)))return 17;
        mouse->Acquire();g_di_kind.clear();g_di_dx=7;
        g_di_mouse_events.clear();g_di_mouse_events.push_back({DIMOFS_BUTTON0,0x80});
        for(int peek=0;peek<2;++peek) {
            count=16;hr=mouse->GetDeviceData(sizeof(events[0]),events,&count,DIGDD_PEEK);
            bool visible=false;for(DWORD i=0;i<count;++i)visible|=events[i].dwOfs==DIMOFS_BUTTON0&&events[i].dwData==0x80;
            if(FAILED(hr)||!visible||g_di_mouse_events.size()!=1||g_di_dx!=7)return 18;
            ++checks;
        }
        count=16;hr=mouse->GetDeviceData(sizeof(events[0]),events,&count,0);
        if(FAILED(hr)||g_di_mouse_events.size()!=0||g_di_dx!=0)return 19;
        ++checks;g_vk[VK_LBUTTON]=0x80;DIMOUSESTATE2 mouse_state{};
        hr=mouse->GetDeviceState(sizeof(mouse_state),&mouse_state);
        if(FAILED(hr)||mouse_state.rgbButtons[0]!=0x80)return 20;
        ++checks;g_vk[VK_LBUTTON]=0;mouse->Unacquire();mouse->Release();root->Release();
    }
    for(auto target:g_hook_targets)MH_DisableHook(target);
    MH_Uninitialize();DestroyWindow(hwnd);FreeLibrary(legacy);
    std::cout<<"{\"passed\":true,\"directinput_checks\":"<<checks<<",\"gpu_used\":false,\"desktop_input_sent\":false}"<<std::endl;
    return 0;
}
