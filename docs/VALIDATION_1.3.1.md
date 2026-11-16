# Phoenix 1.3.1 validation

- The x86 and x64 Phoenix Input package checks load the addon into two processes sharing one directory and verify that only the configured executable owns the mailbox and answers a request.
- Native input tests exercise viewport switching and message forwarding across Unicode and ANSI windows.
- The Windows release checks cover addon loading and embedded dependency extraction. Linux CI builds and checks the Vulkan layer and SDK package. These checks run without game launches or physical GPU inference.
