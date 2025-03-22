#ifndef TETRIS_EXPORT_H
#define TETRIS_EXPORT_H

#ifdef _WIN32
  #ifdef TETRIS_EXPORTS
    #define TETRIS_API __declspec(dllexport)
  #else
    #define TETRIS_API __declspec(dllimport)
  #endif
#else
  #define TETRIS_API
#endif

#endif // TETRIS_EXPORT_H