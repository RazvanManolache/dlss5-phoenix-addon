#pragma once
#include <windows.h>
#include <GL/gl.h>
#include <array>
#include <iostream>
#include "../../src/native/model/bundle_loader.h"

// Test-only graphics/TLS guard shared by the isolated WOW64 probes.
namespace phoenix_wow64_tests {
class OpenGlContext {
    HWND window_=nullptr;
    HDC dc_=nullptr;
    HGLRC context_=nullptr;
    std::array<void*,64> thread_slots_{};
  public:
    ~OpenGlContext() {
        // Preserve test cleanup if an unqualified native loader cleared an
        // existing x86 slot. Never use this restoration in the actual addon.
        for(unsigned i=0;i<thread_slots_.size();++i)
            if(thread_slots_[i]&&!TlsGetValue(i))TlsSetValue(i,thread_slots_[i]);
        if(context_) {wglMakeCurrent(nullptr,nullptr);wglDeleteContext(context_);}
        if(dc_)ReleaseDC(window_,dc_);
        if(window_)DestroyWindow(window_);
    }
    void create() {
        WNDCLASSW type{};type.style=CS_OWNDC;type.lpfnWndProc=DefWindowProcW;
        type.hInstance=GetModuleHandleW(nullptr);type.lpszClassName=L"PhoenixNativeGlProbe";
        phoenix_bundle::require(RegisterClassW(&type)!=0,"OpenGL probe class failed");
        window_=CreateWindowW(type.lpszClassName,L"Phoenix native OpenGL probe",WS_POPUP,
            0,0,64,64,nullptr,nullptr,type.hInstance,nullptr);
        phoenix_bundle::require(window_!=nullptr,"OpenGL probe window failed");
        dc_=GetDC(window_);
        PIXELFORMATDESCRIPTOR format{};format.nSize=sizeof(format);format.nVersion=1;
        format.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER;
        format.iPixelType=PFD_TYPE_RGBA;format.cColorBits=24;
        const auto index=ChoosePixelFormat(dc_,&format);
        phoenix_bundle::require(index&&SetPixelFormat(dc_,index,&format),"OpenGL pixel format failed");
        context_=wglCreateContext(dc_);
        phoenix_bundle::require(context_&&wglMakeCurrent(dc_,context_),"OpenGL context failed");
        const auto renderer=reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        phoenix_bundle::require(renderer&&strstr(renderer,"NVIDIA"),"Probe needs the NVIDIA x86 OpenGL ICD");
        std::cout<<nlohmann::json{{"opengl_renderer",renderer}}.dump()<<std::endl;
        for(unsigned i=0;i<thread_slots_.size();++i)thread_slots_[i]=TlsGetValue(i);
    }
    void render() {
        for(unsigned i=0;i<thread_slots_.size();++i)
            if(thread_slots_[i]&&!TlsGetValue(i)) {
                std::cout<<nlohmann::json{{"cleared_x86_tls_slot",i}}.dump()<<std::endl;
                throw std::runtime_error("Native worker cleared an existing x86 TLS slot "+std::to_string(i));
            }
        glViewport(0,0,64,64);glClearColor(.25f,.5f,.75f,1.f);glClear(GL_COLOR_BUFFER_BIT);
        unsigned char pixel[3]{};glReadPixels(16,16,1,1,GL_RGB,GL_UNSIGNED_BYTE,pixel);
        phoenix_bundle::require(glGetError()==GL_NO_ERROR&&pixel[0]>=62&&pixel[0]<=66&&
            pixel[1]>=126&&pixel[1]<=130&&pixel[2]>=189&&pixel[2]<=193,"OpenGL probe pixels changed");
        phoenix_bundle::require(SwapBuffers(dc_)!=0,"OpenGL probe swap failed");
    }
};

}
