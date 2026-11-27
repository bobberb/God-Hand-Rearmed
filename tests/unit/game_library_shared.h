#pragma once

// Defined in a header, as the runtime's headers define inline functions with statics and inline variables: the test
// program and a fixture library each compile their own copy, and the dynamic linker must make them one object
// (game_library.h: why the library takes them from GHR and is not linked -Bsymbolic).
inline int &sharedCounter()
{
    static int value = 0;
    return value;
}

inline int sharedInlineVariable = 0;
