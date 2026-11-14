// Isolated real-DX9 slot/pairing validation. No ReShade injection or game.
#define NOMINMAX
#include "../../src/integration/compatibility/d3d9_compositor.h"
#include <iostream>
int main()try {
    using namespace phoenix_compat;
    struct Window {
        HWND value=CreateWindowW(L"STATIC",L"Phoenix isolated capture queue",WS_POPUP,
            0,0,64,48,nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);
        ~Window(){if(value)DestroyWindow(value);}
    } window;
    require(window.value!=nullptr,"DX9 probe window");
    Ptr<IDirect3D9> api;api.Attach(Direct3DCreate9(D3D_SDK_VERSION));require(bool(api),"DX9 probe API");
    D3DPRESENT_PARAMETERS pp{};pp.BackBufferWidth=64;pp.BackBufferHeight=48;
    pp.BackBufferFormat=D3DFMT_A8R8G8B8;pp.BackBufferCount=1;pp.SwapEffect=D3DSWAPEFFECT_DISCARD;
    pp.Windowed=TRUE;pp.hDeviceWindow=window.value;pp.PresentationInterval=D3DPRESENT_INTERVAL_IMMEDIATE;
    Ptr<IDirect3DDevice9> device;
    hr(api->CreateDevice(0,D3DDEVTYPE_HAL,window.value,D3DCREATE_HARDWARE_VERTEXPROCESSING|D3DCREATE_FPU_PRESERVE,&pp,&device));
    Ptr<IDirect3DSurface9> back,staging;hr(device->GetBackBuffer(0,0,D3DBACKBUFFER_TYPE_MONO,&back));
    hr(device->CreateOffscreenPlainSurface(64,48,D3DFMT_A8R8G8B8,D3DPOOL_SYSTEMMEM,&staging,nullptr));
    D3D9Compositor compositor(device.Get(),64,48,D3DFMT_A8R8G8B8);
    require(compositor.capture_capacity()>=2,"Probe could not allocate both snapshot slots");
    auto capture=[&](DWORD color,unsigned slot){hr(device->ColorFill(back.Get(),nullptr,color));compositor.capture(back.Get(),slot);};
    auto pixel=[&](unsigned x){D3DLOCKED_RECT mapped{};
        const auto deadline=GetTickCount64()+2000;
        HRESULT result;
        do {
            result=staging->LockRect(&mapped,nullptr,D3DLOCK_READONLY|D3DLOCK_DONOTWAIT);
            if(result==D3DERR_WASSTILLDRAWING)Sleep(1);
        } while(result==D3DERR_WASSTILLDRAWING&&GetTickCount64()<deadline);
        hr(result);
        auto value=static_cast<const DWORD*>(mapped.pBits)[x];hr(staging->UnlockRect());return value;};
    auto read=[&](unsigned slot,DWORD expected){compositor.readback_completed_capture(staging.Get(),slot);
        require(pixel(0)==expected,"Queued DX9 snapshot overwritten or reordered");};
    capture(0xffff0000,0);capture(0xff0000ff,1);
    read(0,0xffff0000);read(1,0xff0000ff);
    compositor.match_completed_input(0);
    capture(0xff00ff00,0);
    Ptr<IDirect3DTexture9> output;Ptr<IDirect3DSurface9> surface;
    hr(device->CreateTexture(64,48,1,D3DUSAGE_RENDERTARGET,D3DFMT_A8R8G8B8,D3DPOOL_DEFAULT,&output,nullptr));
    hr(output->GetSurfaceLevel(0,&surface));
    auto compare=[&](DWORD input,DWORD processed){
        hr(device->ColorFill(surface.Get(),nullptr,processed));
        compositor.draw(back.Get(),output.Get(),2,true);
        hr(device->GetRenderTargetData(back.Get(),staging.Get()));
        require(pixel(8)==input&&pixel(48)==processed,"DX9 comparison lost the submitted input/output pair");};
    compare(0xffff0000,0xffffffff);
    compositor.match_completed_input(1);
    capture(0xffffff00,1);
    compare(0xff0000ff,0xff202020);
    read(0,0xff00ff00);
    compositor.match_completed_input(0);
    compare(0xff00ff00,0xff808080);
    read(1,0xffffff00);
    std::cout<<"{\"passed\":true,\"gpu_used\":true,\"models_run\":false,\"games_run\":false,\"slots\":2,\"paired_inputs\":3}\n";
}catch(const std::exception& e){std::cerr<<e.what();return 1;}
