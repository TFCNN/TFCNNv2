/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
        November 2020 - TFCNNv2.1
--------------------------------------------------
    Tiny Fully Connected Neural Network Library
    https://github.com/tfcnn

    This version uses derivatives w.r.t x rather
    than w.r.t f(x) which is what TFCNNv2 does.
*/

#ifndef TFCNN_H
#define TFCNN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <sys/file.h>

#define uint unsigned int

/*
--------------------------------------
    structures
--------------------------------------
*/

// perceptron struct
struct
{
    float* data;
    float* momentum;
    float bias;
    float bias_momentum;
    uint weights;
}
typedef ptron;

// network struct
struct
{
    // hyperparameters
    uint  init;
    uint  activator;
    uint  optimiser;
    uint  batches;
    float rate;
    float gain;
    float dropout;
    float wdropout;
    float momentum;
    float rmsalpha;
    float epsilon;
    float min_target;
    float max_target;

    // layers
    ptron** layer;

    // count
    uint num_inputs;
    uint num_layers;
    uint num_layerunits;

    // backprop
    uint cbatches;
    float** output;
    float foutput;
    float error;

    // selu alpha dropout
    float drop_a;
    float drop_b;
    float drop_wa;
    float drop_wb;
}
typedef network;

/*
--------------------------------------
    ERROR TYPES
--------------------------------------
*/

#define ERROR_UNINITIALISED_NETWORK -1
#define ERROR_TOOFEWINPUTS -2
#define ERROR_TOOFEWLAYERS -3
#define ERROR_TOOSMALL_LAYERSIZE -4
#define ERROR_ALLOC_LAYERS_ARRAY_FAIL -5
#define ERROR_ALLOC_LAYERS_FAIL -6
#define ERROR_ALLOC_OUTPUTLAYER_FAIL -7
#define ERROR_ALLOC_PERCEPTRON_DATAWEIGHTS_FAIL -8
#define ERROR_ALLOC_PERCEPTRON_ALPHAWEIGHTS_FAIL -9
#define ERROR_CREATE_FIRSTLAYER_FAIL -10
#define ERROR_CREATE_HIDDENLAYER_FAIL -11
#define ERROR_CREATE_OUTPUTLAYER_FAIL -12
#define ERROR_ALLOC_OUTPUT_ARRAY_FAIL -13
#define ERROR_ALLOC_OUTPUT_FAIL -14

/*
--------------------------------------
    DEFINES / ENUMS
--------------------------------------
*/

enum
{
    LEARN_MAX = 1,
    LEARN_MIN = 0,
    NO_LEARN  = -1
}
typedef learn_type;

enum
{
    WEIGHT_INIT_UNIFORM             = 0,
    WEIGHT_INIT_UNIFORM_GLOROT      = 1,
    WEIGHT_INIT_UNIFORM_LECUN       = 2,
    WEIGHT_INIT_UNIFORM_LECUN_POW   = 3,
    WEIGHT_INIT_NORMAL              = 4,
    WEIGHT_INIT_NORMAL_GLOROT       = 5,
    WEIGHT_INIT_NORMAL_LECUN        = 6,
    WEIGHT_INIT_NORMAL_LECUN_POW    = 7
}
typedef weight_init_type;

enum
{
    SELU        = 0,
    GELU        = 1,
    MISH        = 2,
    ISRU        = 3,
    SQNL        = 4,
    SINC        = 5
}
typedef activator;

enum 
{
    OPTIM_SGD       = 0,
    OPTIM_MOMENTUM  = 1,
    OPTIM_NESTEROV  = 2,
    OPTIM_ADAGRAD   = 3,
    OPTIM_RMSPROP   = 4
}
typedef optimiser;

/*
--------------------------------------
    random functions
--------------------------------------
*/

#define FAST_PREDICTABLE_MODE

// quick randoms
float qRandNormal();
float qRandFloat(const float min, const float max);
float qRandWeight(const float min, const float max);
uint  qRand(const uint min, const uint umax);

// slower randoms with higher entropy [make sure FAST_PREDICTABLE_MODE is undefined]
float uRandNormal();
float uRandFloat(const float min, const float max);
float uRandWeight(const float min, const float max);
uint  uRand(const uint min, const uint umax);

// seed with high granularity
void newSRAND();

/*
--------------------------------------
    accessors
--------------------------------------
*/

void setWeightInit(network* net, const weight_init_type u);
void setOptimiser(network* net, const optimiser u);
void setActivator(network* net, const activator u);
void setBatches(network* net, const uint u);
void setLearningRate(network* net, const float f);
void setGain(network* net, const float f);
void setUnitDropout(network* net, const float f); //Dropout
void setWeightDropout(network* net, const float f); //Drop Connect
void setMomentum(network* net, const float f); //SGDM & NAG
void setRMSAlpha(network* net, const float f);
void setEpsilon(network* net, const float f); //ADA & RMS
void setTargetMin(network* net, const float f);
void setTargetMax(network* net, const float f);
void randomHyperparameters(network* net);

/*
--------------------------------------
    neural net functions
--------------------------------------
*/

int createNetwork(network* net, const weight_init_type init_type, const uint num_inputs, const uint num_hidden_layers, const uint num_layer_units, const uint default_settings);
float processNetwork(network* net, const float* inputs, const learn_type learn);
void resetNetwork(network* net);
void destroyNetwork(network* net);
int saveNetwork(network* net, const char* file);
int loadNetwork(network* net, const char* file);

/*
--------------------------------------
    the code ...
--------------------------------------
*/

float qRandFloat(const float min, const float max)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    return ( ( (((float)rand())+1e-7) / (float)RAND_MAX ) * (max-min) ) + min;
}

float uRandFloat(const float min, const float max)
{
#ifdef FAST_PREDICTABLE_MODE
    return qRandFloat(min, max);
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    unsigned int s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    return ( ( (((float)rand())+1e-7) / (float)RAND_MAX ) * (max-min) ) + min;
#endif
}

float qRandWeight(const float min, const float max)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv2 = ( ( (((float)rand())+1e-7) / (float)RAND_MAX ) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
}

float uRandWeight(const float min, const float max)
{
#ifdef FAST_PREDICTABLE_MODE
    return qRandWeight(min, max);
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    unsigned int s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv2 = ( ( (((float)rand())+1e-7) / (float)RAND_MAX ) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
#endif
}

uint qRand(const uint min, const uint umax)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    const uint max = umax + 1;
    return ( ( (((float)rand())+1e-7) / (float)RAND_MAX ) * (max-min) ) + min;
}

uint uRand(const uint min, const uint umax)
{
#ifdef FAST_PREDICTABLE_MODE
    return qRand(min, umax);
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    unsigned int s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    const uint max = umax + 1;
    return ( ( (((float)rand())+1e-7) / (float)RAND_MAX ) * (max-min) ) + min;
#endif
}

float qRandNormal() // Box Muller
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    double u = ( (((float)rand())+1e-7) / (float)RAND_MAX) * 2 - 1;
    double v = ( (((float)rand())+1e-7) / (float)RAND_MAX) * 2 - 1;
    double r = u * u + v * v;
    while(r == 0 || r > 1)
    {
        u = ( (((float)rand())+1e-7) / (float)RAND_MAX) * 2 - 1;
        v = ( (((float)rand())+1e-7) / (float)RAND_MAX) * 2 - 1;
        r = u * u + v * v;
    }
    return u * sqrt(-2 * log(r) / r);
}

float uRandNormal()
{
#ifdef FAST_PREDICTABLE_MODE
    return qRandNormal();
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    unsigned int s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    double u = ( (((float)rand())+1e-7) / (float)RAND_MAX) * 2 - 1;
    double v = ( (((float)rand())+1e-7) / (float)RAND_MAX) * 2 - 1;
    double r = u * u + v * v;
    while(r == 0 || r > 1)
    {
        u = ( (((float)rand())+1e-7) / (float)RAND_MAX) * 2 - 1;
        v = ( (((float)rand())+1e-7) / (float)RAND_MAX) * 2 - 1;
        r = u * u + v * v;
    }
    return u * sqrt(-2 * log(r) / r);
#endif
}

void newSRAND()
{
    struct timespec c;
    clock_gettime(CLOCK_MONOTONIC, &c);
    srand(time(0)+c.tv_nsec);
}

/**********************************************/

static inline float sigmoid(const float x)
{
    return 1 / (1 + exp(-x));
}

static inline float sigmoidDerivative(const float x)
{
    return x * (1 - x);
}

/**********************************************/

static inline float selu(const float x)
{
    if(x < 0){return 1.0507 * (1.67326 * exp(x) - 1.67326);}
    return 1.0507 * x;
}

static inline float seluDerivative(const float x)
{
    if(x < 0){return 1.0507 * (1.67326 * exp(x));}
    return 1.0507;
}

/**********************************************/

static inline float sqnl(const float x)
{
    if(x > 2.0){return 1;}
    else if(x < -2.0){return -1.0;}
    else if(x <= 2.0 && x >= 0){return (x-(x*x))*0.25;}
    return (x+(x*x))*0.25;
}

static inline float sqnlDerivative(const float x)
{
    if(x > 0){return 1 - x * 0.5;}
    return 1 + x * 0.5;
}

/**********************************************/

static inline float gelu(const float x)
{
    return 0.5 * x * (1 + tanh( 0.45015815796 * ( x + 0.044715 * (x*x*x) ) ));
}

static inline float geluDerivative(const float x)
{
    const double x3 = x*x*x;
    const double s2 = 1 / cosh(0.0356774*x3 + 0.797885*x);
    return 0.5 * tanh(0.0356774*x3 + 0.797885*x) + (0.0535161*x3 + 0.398942*x) * (s2 * s2) + 0.5;
}

/**********************************************/

static inline float isru(const float x)
{
    return x / sqrt(1+x*x);
}

static inline float isruDerivative(const float x)
{
    const float x1 = (1 / sqrt(1 + x*x));
    return x1*x1*x1;
}

/**********************************************/

static inline float sinc(const float x)
{
    if(x == 0){return 1;}
    return sin(x) / x;
}

static inline float sincDerivative(const float x)
{
    if(x == 0){return 0;}
    return (cos(x) / x) - (sin(x) / (x*x));
}

/**********************************************/

static inline float mish(const float x)
{
    return x * tanh(log(exp(x) + 1));
}

static inline float mishDerivative(const float x)
{
    const float sech = 1 / cosh(log(exp(x) + 1));
    return tanh(log(exp(x) + 1))   +   (  (x * exp(x) * (sech*sech))  /  (exp(x) + 1)  );
}

/**********************************************/
static inline float Derivative(const float x, const network* net)
{
    if(net->activator == 1)
        return geluDerivative(x);
    else if(net->activator == 2)
        return mishDerivative(x);
    else if(net->activator == 3)
        return isruDerivative(x);
    else if(net->activator == 4)
        return sqnlDerivative(x);
    else if(net->activator == 5)
        return sincDerivative(x);
    
    return seluDerivative(x);
}

static inline float Activator(const float x, const network* net)
{
    if(net->activator == 1)
        return gelu(x);
    else if(net->activator == 2)
        return mish(x);
    else if(net->activator == 3)
        return isru(x);
    else if(net->activator == 4)
        return sqnl(x);
    else if(net->activator == 5)
        return sinc(x);

    return selu(x);
}

/**********************************************/

static inline float SGD(network* net, const float input, const float error)
{
    return net->rate * error * input;
}

static inline float Momentum(network* net, const float input, const float error, float* momentum)
{
    // const float err = (_lrate * error * input);
    // const float ret = err + _lmomentum * momentum[0];
    // momentum[0] = err;
    // return ret;

    const float err = (net->rate * error * input) + net->momentum * momentum[0];
    momentum[0] = err;
    return err;
}

static inline float Nesterov(network* net, const float input, const float error, float* momentum)
{
    const float v = net->momentum * momentum[0] + ( net->rate * error * input );
    const float n = v + net->momentum * (v - momentum[0]);
    momentum[0] = v;
    return n;
}

static inline float ADAGrad(network* net, const float input, const float error, float* momentum)
{
    const float err = error * input;
    momentum[0] += err * err;
    return (net->rate / sqrt(momentum[0] + net->epsilon)) * err;
}

static inline float RMSProp(network* net, const float input, const float error, float* momentum)
{
    const float err = error * input;
    momentum[0] = net->rmsalpha * momentum[0] + (1 - net->rmsalpha) * (err * err);
    return (net->rate / sqrt(momentum[0] + net->epsilon)) * err;
}

static inline float Optimiser(network* net, const float input, const float error, float* momentum)
{
    if(net->optimiser == 1)
        return Momentum(net, input, error, momentum);
    else if(net->optimiser == 2)
        return Nesterov(net, input, error, momentum);
    else if(net->optimiser == 3)
        return ADAGrad(net, input, error, momentum);
    else if(net->optimiser == 4)
        return RMSProp(net, input, error, momentum);
    
    return SGD(net, input, error);
}

/**********************************************/

static inline float doPerceptron(const float* in, ptron* p)
{
    float ro = 0;
    for(uint i = 0; i < p->weights; i++)
        ro += in[i] * p->data[i];
    ro += p->bias;

    return ro;
}

static inline float doDropout(const network* net, const float f, const uint type)
{
    if(type == 0)
    {
        if(net->dropout == 0)
            return f;

        if(uRandFloat(0, 1) <= net->dropout)
        {
            if(net->activator == SELU)
                return f * net->drop_a + net->drop_b;
            else
                return 0;
        }
    }
    else if(type == 1)
    {
        if(net->wdropout == 0)
            return f;

        if(uRandFloat(0, 1) <= net->wdropout)
        {
            if(net->activator == SELU)
                return f * net->drop_wa + net->drop_wb;
            else
                return 0;
        }
    }
    return f;
}

/**********************************************/

int createPerceptron(ptron* p, const uint weights, const float d, const weight_init_type wit)
{
    p->data = malloc(weights * sizeof(float));
    if(p->data == NULL)
        return ERROR_ALLOC_PERCEPTRON_DATAWEIGHTS_FAIL;

    p->momentum = malloc(weights * sizeof(float));
    if(p->momentum == NULL)
    {
        free(p->data);
        return ERROR_ALLOC_PERCEPTRON_ALPHAWEIGHTS_FAIL;
    }

    p->weights = weights;

    for(uint i = 0; i < p->weights; i++)
    {
        if(wit < 4)
            p->data[i] = qRandWeight(-d, d); // uniform
        else
            p->data[i] = qRandNormal() * d;  // normal

        p->momentum[i] = 0;
    }

    p->bias = 0;
    p->bias_momentum = 0;

    return 0;
}

void resetPerceptron(ptron* p, const float d, const weight_init_type wit)
{
    for(uint i = 0; i < p->weights; i++)
    {
        if(wit < 4)
            p->data[i] = qRandWeight(-d, d); // uniform
        else
            p->data[i] = qRandNormal() * d;  // normal
        
        p->momentum[i] = 0;
    }

    p->bias = 0;
    p->bias_momentum = 0;
}

void setWeightInit(network* net, const weight_init_type u)
{
    if(net == NULL){return;}
    net->init = u;
}

void setOptimiser(network* net, const optimiser u)
{
    if(net == NULL){return;}
    net->optimiser = u;
}

void setActivator(network* net, const activator u)
{
    if(net == NULL){return;}
    net->activator = u;
}

void setBatches(network* net, const uint u)
{
    if(net == NULL){return;}
    if(u == 0)
        net->batches = 1;
    else
        net->batches = u;
}

void setLearningRate(network* net, const float f)
{
    if(net == NULL){return;}
    net->rate = f;
}

void setGain(network* net, const float f)
{
    if(net == NULL){return;}
    net->gain = f;
}

void setUnitDropout(network* net, const float f)
{
    if(net == NULL){return;}
    net->dropout = f;

    const float d1 = 1 - net->dropout;
    net->drop_a = pow(net->dropout + 3.090895504 * net->dropout * d1, -0.5);
    net->drop_b = -net->drop_a * (d1 * -1.758094282);
}

void setWeightDropout(network* net, const float f)
{
    if(net == NULL){return;}
    net->wdropout = f;

    const float d1 = 1 - net->wdropout;
    net->drop_wa = pow(net->wdropout + 3.090895504 * net->wdropout * d1, -0.5);
    net->drop_wb = -net->drop_wa * (d1 * -1.758094282);
}

void setMomentum(network* net, const float f)
{
    if(net == NULL){return;}
    net->momentum = f;
}

void setRMSAlpha(network* net, const float f)
{
    if(net == NULL){return;}
    net->rmsalpha = f;
}

void setEpsilon(network* net, const float f)
{
    if(net == NULL){return;}
    net->epsilon = f;
}

void setTargetMin(network* net, const float f)
{
    if(net == NULL){return;}
    net->min_target = f;
}

void setTargetMax(network* net, const float f)
{
    if(net == NULL){return;}
    net->max_target = f;
}

void randomHyperparameters(network* net)
{
    if(net == NULL){return;}
        
    net->init       = uRand(0, 7);
    net->activator  = uRand(0, 5);
    net->optimiser  = uRand(0, 4);
    net->rate       = uRandFloat(0.001, 0.1);
    net->dropout    = uRandFloat(0, 0.99);
    net->wdropout   = uRandFloat(0, 0.99);
    net->momentum   = uRandFloat(0.01, 0.99);
    net->rmsalpha   = uRandFloat(0.01, 0.99);
    net->epsilon    = uRandFloat(1e-8, 1e-5);
}

int createNetwork(network* net, const uint init_weights_type, const uint inputs, const uint hidden_layers, const uint layers_size, const uint default_settings)
{
    const uint layers = hidden_layers+2;

    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(inputs < 1)
        return ERROR_TOOFEWINPUTS;
    if(layers < 3)
        return ERROR_TOOFEWLAYERS;
    if(layers_size < 1)
        return ERROR_TOOSMALL_LAYERSIZE;

    // init net hyper parameters to some default
    net->num_layerunits = layers_size;
    net->num_inputs = inputs;
    net->num_layers = layers;
    net->init       = init_weights_type;
    if(default_settings == 1)
    {
        net->activator  = 0;
        net->optimiser  = 2;
        net->batches    = 3;
        net->rate       = 0.01;
        net->gain       = 1.0;
        net->dropout    = 0.3;
        net->wdropout   = 0;
        net->momentum   = 0.1;
        net->rmsalpha   = 0.2;
        net->epsilon    = 1e-7;
        net->min_target = 0;
        net->max_target = 1;
    }
    net->cbatches   = 0;
    net->error      = 0;
    net->foutput    = 0;

    float d1 = 1 - net->dropout;
    net->drop_a = pow(net->dropout + 3.090895504 * net->dropout * d1, -0.5);
    net->drop_b = -net->drop_a * (d1 * -1.758094282);
    
    d1 = 1 - net->wdropout;
    net->drop_wa = pow(net->wdropout + 3.090895504 * net->wdropout * d1, -0.5);
    net->drop_wb = -net->drop_wa * (d1 * -1.758094282);
    
    // create layers
    net->output = malloc((layers-1) * sizeof(float*));
    if(net->output == NULL)
    {
        destroyNetwork(net);
        return ERROR_ALLOC_OUTPUT_ARRAY_FAIL;
    }
    for(int i = 0; i < layers-1; i++)
    {
        net->output[i] = malloc(layers_size * sizeof(float));
        if(net->output[i] == NULL)
        {
            destroyNetwork(net);
            return ERROR_ALLOC_OUTPUT_FAIL;
        }
    }

    net->layer = malloc(layers * sizeof(ptron*));
    if(net->layer == NULL)
    {
        destroyNetwork(net);
        return ERROR_ALLOC_LAYERS_ARRAY_FAIL;
    }
    for(int i = 0; i < layers-1; i++)
    {
        net->layer[i] = malloc(layers_size * sizeof(ptron));
        if(net->layer[i] == NULL)
        {
            destroyNetwork(net);
            return ERROR_ALLOC_LAYERS_FAIL;
        }
    }

    net->layer[layers-1] = malloc(sizeof(ptron));
    if(net->layer[layers-1] == NULL)
    {
        destroyNetwork(net);
        return ERROR_ALLOC_OUTPUTLAYER_FAIL;
    }

    // init weight
    float d = 1; //WEIGHT_INIT_UNIFORM / WEIGHT_INIT_NORMAL
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(inputs+layers_size));
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrt(3.0/inputs);
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN_POW)
        d = pow(inputs, 0.5);
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(inputs+layers_size));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrt(1.0/inputs);

    // create first layer perceptrons
    for(int i = 0; i < layers_size; i++)
    {
        if(createPerceptron(&net->layer[0][i], inputs, d, net->init) < 0)
        {
            destroyNetwork(net);
            return ERROR_CREATE_FIRSTLAYER_FAIL;
        }
    }
    
    // weight init
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(layers_size+layers_size));
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrt(3.0/layers_size);
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN_POW || init_weights_type == WEIGHT_INIT_NORMAL_LECUN_POW)
        d = pow(layers_size, 0.5);
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(layers_size+layers_size));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrt(1.0/layers_size);

    // create hidden layers
    for(uint i = 1; i < layers-1; i++)
    {
        for(int j = 0; j < layers_size; j++)
        {
            if(createPerceptron(&net->layer[i][j], layers_size, d, net->init) < 0)
            {
                destroyNetwork(net);
                return ERROR_CREATE_HIDDENLAYER_FAIL;
            }
        }
    }

    // weight init
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(layers_size+1));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(layers_size+1));

    // create output layer
    if(createPerceptron(&net->layer[layers-1][0], layers_size, d, net->init) < 0)
    {
        destroyNetwork(net);
        return ERROR_CREATE_OUTPUTLAYER_FAIL;
    }

    // done
    return 0;
}

float processNetwork(network* net, const float* inputs, const learn_type learn)
{
    // validate [it's ok, the output should be sigmoid 0-1 otherwise]
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    
/**************************************
    Forward Prop
**************************************/

    // outputs per layer / unit
    float of[net->num_layers-1][net->num_layerunits];
    float output = 0;

    if(learn == NO_LEARN)
    {
        // input layer
        for(int i = 0; i < net->num_layerunits; i++)
            of[0][i] = Activator(doPerceptron(inputs, &net->layer[0][i]), net);

        // hidden layers
        for(int i = 1; i < net->num_layers-1; i++)
            for(int j = 0; j < net->num_layerunits; j++)
                of[i][j] = Activator(doPerceptron(&of[i-1][0], &net->layer[i][j]), net);

        // binary classifier output layer
        return sigmoid(doPerceptron(&of[net->num_layers-2][0], &net->layer[net->num_layers-1][0]));
    }
    else
    {
        // because now we only have the derivative with respect to x and not f(x).
        float df[net->num_layers-1][net->num_layerunits];

        // input layer
        for(int i = 0; i < net->num_layerunits; i++)
        {
            of[0][i] = doPerceptron(inputs, &net->layer[0][i]);
            df[0][i] = Activator(of[0][i], net);
        }

        // hidden layers
        for(int i = 1; i < net->num_layers-1; i++)
        {
            for(int j = 0; j < net->num_layerunits; j++)
            {
                of[i][j] = doPerceptron(&df[i-1][0], &net->layer[i][j]);
                df[i][j] = Activator(of[i][j], net);
            }
        }

        // binary classifier output layer
        output = sigmoid(doPerceptron(&df[net->num_layers-2][0], &net->layer[net->num_layers-1][0]));
    }

/**************************************
    Backward Prop Error
**************************************/

    // reset accumulators if cbatches has been reset
    if(net->cbatches == 0)
    {
        for(int i = 0; i < net->num_layers-1; i++)
            memset(net->output[i], 0x00, net->num_layerunits * sizeof(float));

        net->foutput = 0;
        net->error = 0;
    }

    // batch accumulation of outputs
    net->foutput += output;
    for(int i = 0; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            net->output[i][j] += of[i][j];

    // accumulate output error
    float eo = net->max_target;
    if(learn == LEARN_MIN)
        eo = net->min_target;
    net->error += eo - output;

    // batching controller
    net->cbatches++;
    if(net->cbatches < net->batches)
    {
        return output;
    }
    else
    {
        // divide accumulators to mean
        net->error /= net->batches;
        net->foutput /= net->batches;

        for(int i = 0; i < net->num_layers-1; i++)
            for(int j = 0; j < net->num_layerunits; j++)
                net->output[i][j] /= net->batches;

        // reset batcher
        net->cbatches = 0;
    }

    // early return if error is 0
    if(net->error == 0)
        return output;

    // define error buffers
    float ef[net->num_layers-1][net->num_layerunits];

    // output (binary classifier) derivative error
    const float eout = net->gain * sigmoidDerivative(net->foutput) * net->error;

    // output 'derivative error layer' of layer before/behind the output layer
    float ler = 0;
    for(int j = 0; j < net->layer[net->num_layers-1][0].weights; j++)
        ler += net->layer[net->num_layers-1][0].data[j] * eout;
    ler += net->layer[net->num_layers-1][0].bias * eout;
    for(int i = 0; i < net->num_layerunits; i++)
        ef[net->num_layers-2][i] = net->gain * Derivative(net->output[net->num_layers-2][i], net) * ler;

    // output derivative error of all other layers
    for(int i = net->num_layers-3; i >= 0; i--)
    {
        // compute total error of layer above w.r.t all weights and units of the above layer
        float ler = 0;
        for(int j = 0; j < net->num_layerunits; j++)
        {
            for(int k = 0; k < net->layer[i+1][j].weights; k++)
                ler += net->layer[i+1][j].data[k] * ef[i+1][j];
            ler += net->layer[i+1][j].bias * ef[i+1][j];
        }
        // propagate that error to into the error variable of each unit of the current layer
        for(int j = 0; j < net->num_layerunits; j++)
            ef[i][j] = net->gain * Derivative(net->output[i][j], net) * ler;
    }

/**************************************
    Update Weights
**************************************/
    
    // update input layer weights
    for(int j = 0; j < net->num_layerunits; j++)
    {
        for(int k = 0; k < net->layer[0][j].weights; k++)
            net->layer[0][j].data[k] += doDropout(net, Optimiser(net, inputs[k], ef[0][j], &net->layer[0][j].momentum[k]), 1);

        net->layer[0][j].bias += doDropout(net, Optimiser(net, 1, ef[0][j], &net->layer[0][j].bias_momentum), 0);
    }

    // update hidden layer weights
    for(int i = 1; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            for(int k = 0; k < net->layer[i][j].weights; k++)
                net->layer[i][j].data[k] += doDropout(net, Optimiser(net, net->output[i-1][j], ef[i][j], &net->layer[i][j].momentum[k]), 1);

            net->layer[i][j].bias += doDropout(net, Optimiser(net, 1, ef[i][j], &net->layer[i][j].bias_momentum), 0);
        }
    }

    // update output layer weights
    for(int j = 0; j < net->layer[net->num_layers-1][0].weights; j++)
        net->layer[net->num_layers-1][0].data[j] += doDropout(net, Optimiser(net, net->output[net->num_layers-2][j], eout, &net->layer[net->num_layers-1][0].momentum[j]), 1);

    net->layer[net->num_layers-1][0].bias += Optimiser(net, 1, eout, &net->layer[net->num_layers-1][0].bias_momentum);

    // done, return forward prop output
    return output;
}

void resetNetwork(network* net)
{
    // validate
    if(net == NULL)
        return;
    if(net->layer == NULL)
        return;

    // reset batching counter
    for(int i = 0; i < net->num_layers-1; i++)
        memset(net->output[i], 0x00, net->num_layerunits * sizeof(float));
    net->cbatches = 0;
    net->foutput = 0;
    net->error = 0;
    
    // init weight
    float d = 1; //WEIGHT_INIT_RANDOM
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(net->num_inputs+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrt(3.0/net->num_inputs);
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN_POW)
        d = pow(net->num_inputs, 0.5);
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(net->num_inputs+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrt(1.0/net->num_inputs);

    // reset first layer perceptrons
    for(int i = 0; i < net->num_layerunits; i++)
        resetPerceptron(&net->layer[0][i], d, net->init);
    
    // weight init
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(net->num_layerunits+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrt(3.0/net->num_layerunits);
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN_POW || net->init == WEIGHT_INIT_NORMAL_LECUN_POW)
        d = pow(net->num_layerunits, 0.5);
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(net->num_layerunits+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrt(1.0/net->num_layerunits);

    // reset hidden layers
    for(uint i = 1; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            resetPerceptron(&net->layer[i][j], d, net->init);

    // weight init
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(net->num_layerunits+1));
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(net->num_layerunits+1));

    // reset output layer
    resetPerceptron(&net->layer[net->num_layers-1][0], d, net->init);
}

void destroyNetwork(network* net)
{
    // validate
    if(net == NULL)
        return;
    if(net->layer == NULL)
        return;

    // free all perceptron data, percepron units and layers
    for(int i = 0; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            free(net->layer[i][j].data);
            free(net->layer[i][j].momentum);
        }
        free(net->layer[i]);
    }
    free(net->layer[net->num_layers-1][0].data);
    free(net->layer[net->num_layers-1][0].momentum);
    free(net->layer[net->num_layers-1]);
    free(net->layer);

    // free output buffers
    for(int i = 0; i < net->num_layers-1; i++)
        free(net->output[i]);
    free(net->output);
}

int saveNetwork(network* net, const char* file)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;

    FILE* f = fopen(file, "wb");
    if(f != NULL)
    {
        if(flock(fileno(f), LOCK_EX) == -1)
            return -1;

        if(fwrite(&net->num_layerunits, 1, sizeof(uint), f) != sizeof(uint))
            return -1;
        
        if(fwrite(&net->num_inputs, 1, sizeof(uint), f) != sizeof(uint))
            return -1;
        
        if(fwrite(&net->num_layers, 1, sizeof(uint), f) != sizeof(uint))
            return -1;
        
        if(fwrite(&net->init, 1, sizeof(uint), f) != sizeof(uint))
            return -1;

        if(fwrite(&net->activator, 1, sizeof(uint), f) != sizeof(uint))
            return -1;

        if(fwrite(&net->optimiser, 1, sizeof(uint), f) != sizeof(uint))
            return -1;

        if(fwrite(&net->batches, 1, sizeof(uint), f) != sizeof(uint))
            return -1;

        ///

        if(fwrite(&net->rate, 1, sizeof(float), f) != sizeof(float))
            return -1;

        if(fwrite(&net->gain, 1, sizeof(float), f) != sizeof(float))
            return -1;

        if(fwrite(&net->dropout, 1, sizeof(float), f) != sizeof(float))
            return -1;

        if(fwrite(&net->wdropout, 1, sizeof(float), f) != sizeof(float))
            return -1;

        if(fwrite(&net->momentum, 1, sizeof(float), f) != sizeof(float))
            return -1;

        if(fwrite(&net->rmsalpha, 1, sizeof(float), f) != sizeof(float))
            return -1;
        
        if(fwrite(&net->epsilon, 1, sizeof(float), f) != sizeof(float))
            return -1;

        if(fwrite(&net->min_target, 1, sizeof(float), f) != sizeof(float))
            return -1;

        if(fwrite(&net->max_target, 1, sizeof(float), f) != sizeof(float))
            return -1;

        for(int i = 0; i < net->num_layers-1; i++)
        {
            for(int j = 0; j < net->num_layerunits; j++)
            {
                if(fwrite(&net->layer[i][j].data[0], 1, net->layer[i][j].weights*sizeof(float), f) != net->layer[i][j].weights*sizeof(float))
                    return -1;
                
                if(fwrite(&net->layer[i][j].momentum[0], 1, net->layer[i][j].weights*sizeof(float), f) != net->layer[i][j].weights*sizeof(float))
                    return -1;

                if(fwrite(&net->layer[i][j].bias, 1, sizeof(float), f) != sizeof(float))
                    return -1;
                
                if(fwrite(&net->layer[i][j].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                    return -1;
            }
        }

        if(fwrite(&net->layer[net->num_layers-1][0].data[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(float), f) != net->layer[net->num_layers-1][0].weights*sizeof(float))
            return -1;
        
        if(fwrite(&net->layer[net->num_layers-1][0].momentum[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(float), f) != net->layer[net->num_layers-1][0].weights*sizeof(float))
            return -1;

        if(fwrite(&net->layer[net->num_layers-1][0].bias, 1, sizeof(float), f) != sizeof(float))
            return -1;
        
        if(fwrite(&net->layer[net->num_layers-1][0].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            return -1;

        if(flock(fileno(f), LOCK_UN) == -1)
            return -1;

        fclose(f);
    }

    return 0;
}

int loadNetwork(network* net, const char* file)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    
    FILE* f = fopen(file, "rb");
    if(f == NULL)
        return -1;

    ///

    destroyNetwork(net);

    ///

    while(fread(&net->num_layerunits, 1, sizeof(uint), f) != sizeof(uint))
        return -1;

    while(fread(&net->num_inputs, 1, sizeof(uint), f) != sizeof(uint))
        return -1;

    while(fread(&net->num_layers, 1, sizeof(uint), f) != sizeof(uint))
        return -1;

    while(fread(&net->init, 1, sizeof(uint), f) != sizeof(uint))
        return -1;

    while(fread(&net->activator, 1, sizeof(uint), f) != sizeof(uint))
        return -1;

    while(fread(&net->optimiser, 1, sizeof(uint), f) != sizeof(uint))
        return -1;

    while(fread(&net->batches, 1, sizeof(uint), f) != sizeof(uint))
        return -1;

    ///

    while(fread(&net->rate, 1, sizeof(float), f) != sizeof(float))
        return -1;

    while(fread(&net->gain, 1, sizeof(float), f) != sizeof(float))
        return -1;

    while(fread(&net->dropout, 1, sizeof(float), f) != sizeof(float))
        return -1;

    while(fread(&net->wdropout, 1, sizeof(float), f) != sizeof(float))
        return -1;

    while(fread(&net->momentum, 1, sizeof(float), f) != sizeof(float))
        return -1;

    while(fread(&net->rmsalpha, 1, sizeof(float), f) != sizeof(float))
        return -1;
    
    while(fread(&net->epsilon, 1, sizeof(float), f) != sizeof(float))
        return -1;

    while(fread(&net->min_target, 1, sizeof(float), f) != sizeof(float))
        return -1;

    while(fread(&net->max_target, 1, sizeof(float), f) != sizeof(float))
        return -1;

    ///

    createNetwork(net, net->init, net->num_inputs, net->num_layers-2, net->num_layerunits, 0);

    ///

    for(int i = 0; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            while(fread(&net->layer[i][j].data[0], 1, net->layer[i][j].weights*sizeof(float), f) != net->layer[i][j].weights*sizeof(float))
                return -1;

            while(fread(&net->layer[i][j].momentum[0], 1, net->layer[i][j].weights*sizeof(float), f) != net->layer[i][j].weights*sizeof(float))
                return -1;

            while(fread(&net->layer[i][j].bias, 1, sizeof(float), f) != sizeof(float))
                return -1;

            while(fread(&net->layer[i][j].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                return -1;
        }
    }

    ///

    while(fread(&net->layer[net->num_layers-1][0].data[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(float), f) != net->layer[net->num_layers-1][0].weights*sizeof(float))
        return -1;

    while(fread(&net->layer[net->num_layers-1][0].momentum[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(float), f) != net->layer[net->num_layers-1][0].weights*sizeof(float))
        return -1;

    while(fread(&net->layer[net->num_layers-1][0].bias, 1, sizeof(float), f) != sizeof(float))
        return -1;

    while(fread(&net->layer[net->num_layers-1][0].bias_momentum, 1, sizeof(float), f) != sizeof(float))
        return -1;

    ///

    fclose(f);
    return 0;
}

#endif
