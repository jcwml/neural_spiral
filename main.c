// github.com/jcwml

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <X11/Xlib.h>

Display *d;
Window w;
GC gc;
unsigned int ww = 1024, wh = 768, ww2 = 0, wh2 = 0;
unsigned int white, black = 16711680, green = 65280, red = 16711680, grey = 2565927;

unsigned int max_iter = 33;
float rmi = 0.f;

unsigned int neural_skip = 1;

typedef struct
{
    float m;
    unsigned int x;
    unsigned int y;
} pred;
pred predictions[8192];

unsigned int quantise_float(float f)
{
    if(f < 0.f)
        f -= 0.5f;
    else
        f += 0.5f;
    return (unsigned int)f;
}

void loadPredictions(const char* fn)
{
    FILE* f = fopen(fn, "r");
    if(f)
    {
        char line[256];
        unsigned int fni = 0;
        while(fgets(line, 256, f) != NULL)
        {
            float m,x,y;
            if(sscanf(line, "%f,%f,%f", &m, &x, &y) == 3)
            {
                predictions[fni].m = m;
                predictions[fni].x = quantise_float(x);
                predictions[fni].y = quantise_float(y);
                fni++;
                if(fni >= 8192){break;}
            }
        }
        fclose(f);
    }
}

static inline void sfunc(int* x, int* y, float m)
{
    const float sr = 333.f;
    const float st = 33.f;
    *x = sinf(m*st)*sr*m;
    *y = cosf(m*st)*sr*m;
}

void draw()
{
    XClearWindow(d, w);

    XSetForeground(d, gc, grey);
    XDrawLine(d, w, gc, 0, wh2, ww-1, wh2);
    XDrawLine(d, w, gc, ww2, 0, ww2, wh-1);

    // actual function
    int lx=0, ly=0;
    for(int i = 0; i < max_iter; i++)
    {
        int x, y;
        const float m = ((float)i)*rmi;
        sfunc(&x, &y, m);
        x += ww2;
        y += wh2;
        //printf("%i %i %f\n", x, y, m);
        
        XSetForeground(d, gc, red);
        if(i != 0)
            XDrawLine(d, w, gc, lx, ly, x, y);

        XSetForeground(d, gc, white);
        XDrawArc(d, w, gc, x-1, y-1, 3, 3, 360*64, 360*64);

        lx = x, ly = y;
    }

    // neural
    XSetForeground(d, gc, green);
    lx=0, ly=0;
    for(int i = 0; i < 8192; i+=neural_skip)
    {
        const int x = predictions[i].x + ww2;
        const int y = predictions[i].y + wh2;
        //printf("%i %i %f\n", predictions[i].x, predictions[i].y, predictions[i].m);
        
        if(i != 0)
            XDrawLine(d, w, gc, lx, ly, x, y);

        XDrawArc(d, w, gc, x-1, y-1, 3, 3, 360*64, 360*64);

        lx = x, ly = y;
    }
}

int main(int ac, char** av)
{
    // printf("Hello from TensorFlow C library version %s\n", TF_Version());

    if(ac >= 2)
    {
        max_iter = atoi(av[1]);
        if(max_iter > 1024)
            max_iter = 1024;
    }
    rmi = 1.f / ((float)max_iter);

    if(ac >= 3)
        loadPredictions(av[2]);

    if(ac >= 4)
    {
        neural_skip = 8192 / atoi(av[3]);
        if(neural_skip > 1024)
            neural_skip = 1024;
        else if(neural_skip == 0)
            neural_skip = 1;
    }

    d = XOpenDisplay(NULL);
    if(d == NULL){return 0;}

    int s = DefaultScreen(d);
    
    white = WhitePixel(d, s);
    black = BlackPixel(d, s);

    ww2 = ww/2, wh2 = wh/2;
    w = XCreateSimpleWindow(d, DefaultRootWindow(d), 0, 0, 1024, 768, 0, white, black);
    gc = XCreateGC(d, w, 0, NULL);
    Atom wmDel = XInternAtom(d, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(d, w, &wmDel, 1);
    XSelectInput(d, w, StructureNotifyMask);
    XMapWindow(d, w);

    XEvent e;
    do{XNextEvent(d, &e);}while(e.type != MapNotify);
    draw();

    while(1)
    {
        XNextEvent(d, &e);

        if(e.type == ConfigureNotify)
        {
            if(e.xconfigure.width != ww || e.xconfigure.height != wh)
            {
                ww = e.xconfigure.width;
                wh = e.xconfigure.height;
                ww2 = ww/2, wh2 = wh/2;
                draw();
            }
        }
        else if(e.type == ClientMessage && e.xclient.data.l[0] == wmDel)
            break;
    } 

    XDestroyWindow(d, w);
    XCloseDisplay(d);

    return 0;
}

