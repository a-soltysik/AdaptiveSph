#pragma once

#ifndef PD_MOVE_ONLY
#    define PD_MOVE_ONLY(className)                            \
        className(const className&) = delete;                  \
        className(className&&) = default;                      \
        auto operator=(const className&)->className& = delete; \
        auto operator=(className&&)->className& = default
#endif

#ifndef PD_COPY_ONLY
#    define PD_COPY_ONLY(className)                             \
        className(const className&) = default;                  \
        className(className&&) = delete;                        \
        auto operator=(const className&)->className& = default; \
        auto operator=(className&&)->className& = delete
#endif

#ifndef PD_DEFAULT_ALL
#    define PD_DEFAULT_ALL(className)                           \
        className(const className&) = default;                  \
        className(className&&) = default;                       \
        auto operator=(const className&)->className& = default; \
        auto operator=(className&&)->className& = default
#endif

#ifndef PD_DELETE_ALL
#    define PD_DELETE_ALL(className)                           \
        className(const className&) = delete;                  \
        className(className&&) = delete;                       \
        auto operator=(const className&)->className& = delete; \
        auto operator=(className&&)->className& = delete
#endif
