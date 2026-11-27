// Every header the game code compiles against, in one object, as the split build's game code sees them
// (PS2X_SHARED_STATE, PS2X_GAME_CODE), with -fkeep-inline-functions so that every inline function is emitted, and with
// it every static local. tools/linux/check_shared_symbols.py then lists the object's shared data symbols (one copy
// per module on Windows) against tools/linux/shared_state_allowlist.txt. Built by src/CMakeLists.txt (tests only).
//
// The includes: what runtime/ps2xRecomp writes at the top of the generated code (ps2_recompiler.cpp
// writeCombinedOutputPreamble), what the room modules include (module_registry.h), and what the VU1 code compiled
// ahead of time includes (ps2_vu1_impl.inl).

#include <stdexcept>
#include <ps2_recompiled_functions.h>

#include "ps2_runtime_macros.h"
#include "ps2_runtime.h"
#include <ps2_recompiled_stubs.h>
#include "ps2_syscalls.h"
#include "ps2_stubs.h"
#include "ps2_log.h"
#include "ps2_cg.h"
#include "ps2_module_table.h"
#include "kamite/module_registry.h"
#include "ps2_vu1_impl.inl"
