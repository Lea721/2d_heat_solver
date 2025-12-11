#include "visualizer_opengl.h"
#include <GL/freeglut.h>
#include <mutex>
#include <atomic>
#include <vector>
#include <cmath>

using namespace std;

namespace visualizer {

static int nx = 0, ny = 0;
static int win_w = 800, win_h = 800;
static vector<float> img; // floats 0..1 for colors
static mutex grid_mutex;
static atomic<bool> updated(false);

// map value to color (blue -> red)
static void scalar_to_rgb(float v, float &r, float &g, float &b) {
    if (v < 0.f) v = 0.f;
    if (v > 1.f) v = 1.f;
    r = std::min(1.f, 2.f * v);
    b = std::min(1.f, 2.f * (1.f - v));
    g = 1.0f - fabs(2.f * v - 1.f) * 0.8f;
}

// ============================
// FIXED DISPLAY FUNCTION
// ============================
static void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    grid_mutex.lock();

    if (!img.empty()) {
        glPointSize(1.0f);
        glBegin(GL_POINTS);

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {

                int idx = (j * nx + i) * 3;
                glColor3f(img[idx], img[idx+1], img[idx+2]);

                // ---- FIXED COORDINATE MAPPING ----
                float x = (2.0f * i) / (nx - 1) - 1.0f;          // 0..nx-1 → -1..+1
                float y = 1.0f - (2.0f * j) / (ny - 1);          // invert Y axis

                glVertex2f(x, y);
            }
        }

        glEnd();
    }

    grid_mutex.unlock();
    glutSwapBuffers();

    updated.store(false);
}

void init(int argc, char** argv, int grid_nx, int grid_ny, int window_w, int window_h) {
    nx = grid_nx; ny = grid_ny;
    win_w = window_w; win_h = window_h;
    img.clear();
    img.resize(nx * ny * 3, 0.0f);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(win_w, win_h);
    glutCreateWindow("MPI Heatmap Visualizer");

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);

    glutDisplayFunc(display);

    glClearColor(0, 0, 0, 1);
}

void update_grid(const vector<double>& data) {
    if ((int)data.size() != nx * ny) return;

    grid_mutex.lock();
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double v = data[j * nx + i];
            float vn = (float)(v / 100.0);  
            float r, g, b;
            scalar_to_rgb(vn, r, g, b);
            int idx = (j * nx + i) * 3;
            img[idx] = r;
            img[idx+1] = g;
            img[idx+2] = b;
        }
    }
    updated.store(true);
    grid_mutex.unlock();

    glutPostRedisplay();
}

void poll_and_draw() {
    glutMainLoopEvent();
    glutPostRedisplay();
}

void finish() {}

} // namespace visualizer

