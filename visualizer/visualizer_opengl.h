#ifndef VISUALIZER_OPENGL_H
#define VISUALIZER_OPENGL_H

#include <vector>

namespace visualizer {
  // initialize GLUT window (call on rank 0 before time loop)
  void init(int argc, char** argv, int grid_nx, int grid_ny, int window_w=800, int window_h=800);

  // update grid data (call every time you gathered full global grid)
  // data must be exactly grid_nx * grid_ny doubles (row-major)
  void update_grid(const std::vector<double>& data);

  // Called inside your main solver loop on rank 0 to process events and draw
  // This is non-blocking (uses glutMainLoopEvent)
  void poll_and_draw();

  // close/destroy
  void finish();
}

#endif
