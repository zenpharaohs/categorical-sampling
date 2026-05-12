/*
 * Compatibility glue for Apple Clang OpenMP codegen against MATLAB's bundled
 * libomp on macOS.
 *
 * Xcode 26's Clang can emit __kmpc_dispatch_deinit for dynamic schedules.
 * MATLAB R2026a's bundled libomp does not export that newer host-runtime hook,
 * but MATLAB also preloads its own OpenMP runtime. Linking Homebrew libomp
 * instead can abort MATLAB with duplicate OpenMP runtime initialization.
 *
 * Older host libomp runtimes complete dynamic worksharing through the existing
 * dispatch fini calls, so this hook is intentionally a no-op.
 */
#if defined(__APPLE__)
void __kmpc_dispatch_deinit(void *loc, int gtid)
{
    (void)loc;
    (void)gtid;
}
#endif
