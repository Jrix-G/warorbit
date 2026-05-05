#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>

#define BOARD_SIZE 100.0
#define CENTER 50.0
#define SUN_RADIUS 10.0
#define ROTATION_RADIUS_LIMIT 50.0
#define COMET_RADIUS 1.0

typedef struct {
    double x;
    double y;
} Point;

typedef struct {
    long id;
    double x;
    double y;
    double radius;
} Planet;

static double distance_xy(double ax, double ay, double bx, double by) {
    double dx = ax - bx;
    double dy = ay - by;
    return sqrt(dx * dx + dy * dy);
}

static int id_in_list(long id, const long *ids, Py_ssize_t n_ids) {
    for (Py_ssize_t i = 0; i < n_ids; ++i) {
        if (ids[i] == id) {
            return 1;
        }
    }
    return 0;
}

static int item_as_double(PyObject *seq, Py_ssize_t index, double *out) {
    PyObject *item = PySequence_GetItem(seq, index);
    if (item == NULL) {
        return 0;
    }
    *out = PyFloat_AsDouble(item);
    Py_DECREF(item);
    return !PyErr_Occurred();
}

static int item_as_long(PyObject *seq, Py_ssize_t index, long *out) {
    PyObject *item = PySequence_GetItem(seq, index);
    if (item == NULL) {
        return 0;
    }
    *out = PyLong_AsLong(item);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        double value = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(item);
            return 0;
        }
        *out = (long)value;
    }
    Py_DECREF(item);
    return 1;
}

static int uniform_double(PyObject *rng, double low, double high, double *out) {
    PyObject *value = PyObject_CallMethod(rng, "uniform", "dd", low, high);
    if (value == NULL) {
        return 0;
    }
    *out = PyFloat_AsDouble(value);
    Py_DECREF(value);
    return !PyErr_Occurred();
}

static PyObject *point_list(double x, double y) {
    PyObject *pair = PyList_New(2);
    if (pair == NULL) {
        return NULL;
    }
    PyObject *px = PyFloat_FromDouble(x);
    PyObject *py = PyFloat_FromDouble(y);
    if (px == NULL || py == NULL) {
        Py_XDECREF(px);
        Py_XDECREF(py);
        Py_DECREF(pair);
        return NULL;
    }
    PyList_SET_ITEM(pair, 0, px);
    PyList_SET_ITEM(pair, 1, py);
    return pair;
}

static int append_point(PyObject *path, double x, double y) {
    PyObject *pair = point_list(x, y);
    if (pair == NULL) {
        return 0;
    }
    int ok = PyList_Append(path, pair) == 0;
    Py_DECREF(pair);
    return ok;
}

static PyObject *build_paths(const Point *visible, Py_ssize_t n_visible) {
    PyObject *paths = PyList_New(4);
    if (paths == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < 4; ++i) {
        PyObject *path = PyList_New(0);
        if (path == NULL) {
            Py_DECREF(paths);
            return NULL;
        }
        PyList_SET_ITEM(paths, i, path);
    }

    PyObject *p0 = PyList_GET_ITEM(paths, 0);
    PyObject *p1 = PyList_GET_ITEM(paths, 1);
    PyObject *p2 = PyList_GET_ITEM(paths, 2);
    PyObject *p3 = PyList_GET_ITEM(paths, 3);

    for (Py_ssize_t i = 0; i < n_visible; ++i) {
        double x = visible[i].x;
        double y = visible[i].y;
        if (!append_point(p0, y, x) ||
            !append_point(p1, BOARD_SIZE - x, y) ||
            !append_point(p2, x, BOARD_SIZE - y) ||
            !append_point(p3, BOARD_SIZE - y, BOARD_SIZE - x)) {
            Py_DECREF(paths);
            return NULL;
        }
    }
    return paths;
}

static PyObject *generate_comet_paths_c(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *initial_planets_obj = NULL;
    PyObject *comet_planet_ids_obj = Py_None;
    PyObject *rng = NULL;
    double angular_velocity = 0.0;
    long spawn_step = 0;
    double comet_speed = 4.0;

    static char *kwlist[] = {
        "initial_planets",
        "angular_velocity",
        "spawn_step",
        "comet_planet_ids",
        "comet_speed",
        "rng",
        NULL,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "Odl|OdO:generate_comet_paths",
            kwlist,
            &initial_planets_obj,
            &angular_velocity,
            &spawn_step,
            &comet_planet_ids_obj,
            &comet_speed,
            &rng)) {
        return NULL;
    }
    if (rng == NULL || rng == Py_None) {
        PyErr_SetString(PyExc_ValueError, "rng must be provided");
        return NULL;
    }

    PyObject *empty_ids = NULL;
    PyObject *ids_source = comet_planet_ids_obj;
    if (comet_planet_ids_obj == Py_None) {
        empty_ids = PyTuple_New(0);
        if (empty_ids == NULL) {
            return NULL;
        }
        ids_source = empty_ids;
    }
    PyObject *ids_fast = PySequence_Fast(ids_source, "comet_planet_ids must be iterable");
    Py_XDECREF(empty_ids);
    if (ids_fast == NULL) {
        return NULL;
    }
    Py_ssize_t n_ids = PySequence_Fast_GET_SIZE(ids_fast);
    long *ids = NULL;
    if (n_ids > 0) {
        ids = (long *)PyMem_Calloc((size_t)n_ids, sizeof(long));
        if (ids == NULL) {
            Py_DECREF(ids_fast);
            return PyErr_NoMemory();
        }
        for (Py_ssize_t i = 0; i < n_ids; ++i) {
            PyObject *item = PySequence_Fast_GET_ITEM(ids_fast, i);
            ids[i] = PyLong_AsLong(item);
            if (PyErr_Occurred()) {
                PyErr_Clear();
                double value = PyFloat_AsDouble(item);
                if (PyErr_Occurred()) {
                    PyMem_Free(ids);
                    Py_DECREF(ids_fast);
                    return NULL;
                }
                ids[i] = (long)value;
            }
        }
    }
    Py_DECREF(ids_fast);

    PyObject *planets_fast = PySequence_Fast(initial_planets_obj, "initial_planets must be iterable");
    if (planets_fast == NULL) {
        PyMem_Free(ids);
        return NULL;
    }
    Py_ssize_t n_planets = PySequence_Fast_GET_SIZE(planets_fast);
    Planet *static_planets = (Planet *)PyMem_Calloc((size_t)n_planets, sizeof(Planet));
    Planet *orbiting_planets = (Planet *)PyMem_Calloc((size_t)n_planets, sizeof(Planet));
    if (static_planets == NULL || orbiting_planets == NULL) {
        Py_XDECREF(planets_fast);
        PyMem_Free(ids);
        PyMem_Free(static_planets);
        PyMem_Free(orbiting_planets);
        return PyErr_NoMemory();
    }
    Py_ssize_t n_static = 0;
    Py_ssize_t n_orbiting = 0;

    for (Py_ssize_t i = 0; i < n_planets; ++i) {
        PyObject *planet = PySequence_Fast_GET_ITEM(planets_fast, i);
        long id = 0;
        double x = 0.0, y = 0.0, radius = 0.0;
        if (!item_as_long(planet, 0, &id) ||
            !item_as_double(planet, 2, &x) ||
            !item_as_double(planet, 3, &y) ||
            !item_as_double(planet, 4, &radius)) {
            Py_DECREF(planets_fast);
            PyMem_Free(ids);
            PyMem_Free(static_planets);
            PyMem_Free(orbiting_planets);
            return NULL;
        }
        if (id_in_list(id, ids, n_ids)) {
            continue;
        }
        Planet parsed = {id, x, y, radius};
        double pr = distance_xy(x, y, CENTER, CENTER);
        if (pr + radius < ROTATION_RADIUS_LIMIT) {
            orbiting_planets[n_orbiting++] = parsed;
        } else {
            static_planets[n_static++] = parsed;
        }
    }
    Py_DECREF(planets_fast);

    Point *dense = (Point *)PyMem_Calloc(5000, sizeof(Point));
    Point *path = (Point *)PyMem_Calloc(5000, sizeof(Point));
    if (dense == NULL || path == NULL) {
        PyMem_Free(ids);
        PyMem_Free(static_planets);
        PyMem_Free(orbiting_planets);
        PyMem_Free(dense);
        PyMem_Free(path);
        return PyErr_NoMemory();
    }

    PyObject *result = Py_None;
    Py_INCREF(Py_None);

    for (int attempt = 0; attempt < 300; ++attempt) {
        double e = 0.0, a = 0.0, phi = 0.0;
        if (!uniform_double(rng, 0.75, 0.93, &e) ||
            !uniform_double(rng, 60.0, 150.0, &a)) {
            Py_DECREF(result);
            result = NULL;
            goto cleanup;
        }

        double perihelion = a * (1.0 - e);
        if (perihelion < SUN_RADIUS + COMET_RADIUS) {
            continue;
        }

        double b = a * sqrt(1.0 - e * e);
        double c_val = a * e;
        if (!uniform_double(rng, M_PI / 6.0, M_PI / 3.0, &phi)) {
            Py_DECREF(result);
            result = NULL;
            goto cleanup;
        }

        double cos_phi = cos(phi);
        double sin_phi = sin(phi);
        for (int i = 0; i < 5000; ++i) {
            double t = 0.3 * M_PI + 1.4 * M_PI * (double)i / 4999.0;
            double ex = c_val + a * cos(t);
            double ey = b * sin(t);
            dense[i].x = CENTER + ex * cos_phi - ey * sin_phi;
            dense[i].y = CENTER + ex * sin_phi + ey * cos_phi;
        }

        Py_ssize_t path_len = 1;
        path[0] = dense[0];
        double cum = 0.0;
        double target = comet_speed;
        for (int i = 1; i < 5000; ++i) {
            cum += distance_xy(dense[i].x, dense[i].y, dense[i - 1].x, dense[i - 1].y);
            if (cum >= target) {
                path[path_len++] = dense[i];
                target += comet_speed;
            }
        }

        Py_ssize_t board_start = -1;
        Py_ssize_t board_end = -1;
        for (Py_ssize_t i = 0; i < path_len; ++i) {
            double x = path[i].x;
            double y = path[i].y;
            if (0.0 <= x && x <= BOARD_SIZE && 0.0 <= y && y <= BOARD_SIZE) {
                if (board_start < 0) {
                    board_start = i;
                }
                board_end = i;
            }
        }
        if (board_start < 0) {
            continue;
        }
        Py_ssize_t n_visible = board_end - board_start + 1;
        if (!(5 <= n_visible && n_visible <= 40)) {
            continue;
        }
        Point *visible = path + board_start;

        int valid = 1;
        double buf = COMET_RADIUS + 0.5;
        for (Py_ssize_t k = 0; k < n_visible && valid; ++k) {
            double cx = visible[k].x;
            double cy = visible[k].y;
            if (distance_xy(cx, cy, CENTER, CENTER) < SUN_RADIUS + COMET_RADIUS) {
                valid = 0;
                break;
            }

            double sx[4] = {cy, BOARD_SIZE - cx, cx, BOARD_SIZE - cy};
            double sy[4] = {cx, cy, BOARD_SIZE - cy, BOARD_SIZE - cx};

            for (Py_ssize_t p = 0; p < n_static && valid; ++p) {
                Planet planet = static_planets[p];
                for (int s = 0; s < 4; ++s) {
                    if (distance_xy(sx[s], sy[s], planet.x, planet.y) < planet.radius + buf) {
                        valid = 0;
                        break;
                    }
                }
            }
            if (!valid) {
                break;
            }

            double game_step = (double)(spawn_step - 1) + (double)k;
            for (Py_ssize_t p = 0; p < n_orbiting && valid; ++p) {
                Planet planet = orbiting_planets[p];
                double dx = planet.x - CENTER;
                double dy = planet.y - CENTER;
                double orb_r = sqrt(dx * dx + dy * dy);
                double init_angle = atan2(dy, dx);
                double cur_angle = init_angle + angular_velocity * game_step;
                double px = CENTER + orb_r * cos(cur_angle);
                double py = CENTER + orb_r * sin(cur_angle);
                for (int s = 0; s < 4; ++s) {
                    if (distance_xy(sx[s], sy[s], px, py) < planet.radius + COMET_RADIUS) {
                        valid = 0;
                        break;
                    }
                }
            }
        }

        if (valid) {
            PyObject *paths = build_paths(visible, n_visible);
            if (paths == NULL) {
                Py_DECREF(result);
                result = NULL;
                goto cleanup;
            }
            Py_DECREF(result);
            result = paths;
            break;
        }
    }

cleanup:
    PyMem_Free(ids);
    PyMem_Free(static_planets);
    PyMem_Free(orbiting_planets);
    PyMem_Free(dense);
    PyMem_Free(path);
    return result;
}

static PyMethodDef OrbitWarsCMethods[] = {
    {
        "generate_comet_paths",
        (PyCFunction)generate_comet_paths_c,
        METH_VARARGS | METH_KEYWORDS,
        "C implementation of orbit_wars.generate_comet_paths.",
    },
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef orbit_wars_c_module = {
    PyModuleDef_HEAD_INIT,
    "orbit_wars_c",
    "C accelerators for local Orbit Wars simulation.",
    -1,
    OrbitWarsCMethods,
};

PyMODINIT_FUNC PyInit_orbit_wars_c(void) {
    return PyModule_Create(&orbit_wars_c_module);
}
