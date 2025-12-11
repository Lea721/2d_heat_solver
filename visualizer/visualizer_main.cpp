#include "visualizer_opengl.h"
#include <vector>
#include <unistd.h>
#include <cmath>

int main(int argc, char** argv) {
    int nx = 200, ny = 200;

    // Initialize window
    visualizer::init(argc, argv, nx, ny, 800, 800);

    // Fake animation data
    std::vector<double> data(nx * ny);
    
    for (int frame = 0; frame < 500; frame++) {
        for (int i = 0; i < nx * ny; i++) {
            data[i] = 50 + 50 * sin(0.01 * frame + i * 0.0005);
        }

        visualizer::update_grid(data);
        visualizer::poll_and_draw();

        usleep(16000); // ~60 FPS
    }

    visualizer::finish();
    return 0;
}

