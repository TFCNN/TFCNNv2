/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
        October 2020 - TFCNNv2
--------------------------------------------------
    Tiny Fully Connected Neural Network Library
    https://github.com/tfcnn
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
    float dropout_decay;
    float momentum;
    float rmsalpha;
    float elualpha;
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
    IDENTITY    = 0,
    ATAN        = 1,
    TANH        = 2,
    ELU         = 3,
    LEAKYRELU   = 4,
    RELU        = 5,
    SIGMOID     = 6,
    SWISH       = 7,
    LECUN       = 8,
    ELLIOT      = 9,
    SOFTPLUS    = 10,
    GELU        = 11,
    SELU        = 12
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
float qRandFloat(const float min, const float max);
float qRandWeight(const float min, const float max);
uint  qRand(const uint min, const uint umax);

// slower randoms with higher entropy [make sure FAST_PREDICTABLE_MODE is undefined]
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
void setDropoutDecay(network* net, const float f); //Dropout now silences the unit activation by decay rather than on/off
void setMomentum(network* net, const float f); //SGDM & NAG
void setRMSAlpha(network* net, const float f);
void setELUAlpha(network* net, const float f); //ELU & LeakyReLU
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
float processNetwork(network* net, float* inputs, const learn_type learn);
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
    const float rv = (float)rand();
    if(rv == 0)
        return min;
    return ( (rv / (float)RAND_MAX) * (max-min) ) + min;
}

float uRandFloat(const float min, const float max)
{
#ifdef FAST_PREDICTABLE_MODE
    return qRandFloat(min, max);
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    const float rv = (float)rand();
    if(rv == 0)
        return min;
    return ( (rv / (float)RAND_MAX) * (max-min) ) + min;
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
        const float rv = (float)rand();
        if(rv == 0)
            return min;
        const float rv2 = ( (rv / (float)RAND_MAX) * (max-min) ) + min;
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
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv = (float)rand();
        if(rv == 0)
            return min;
        const float rv2 = ( (rv / (float)RAND_MAX) * (max-min) ) + min;
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
    const int rv = rand();
    const uint max = umax + 1;
    if(rv == 0)
        return min;
    return ( ((float)rv / (float)RAND_MAX) * (max-min) ) + min;
}

uint uRand(const uint min, const uint umax)
{
#ifdef FAST_PREDICTABLE_MODE
    return qRand(min, umax);
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    const int rv = rand();
    const uint max = umax + 1;
    if(rv == 0)
        return min;
    return ( ((float)rv / (float)RAND_MAX) * (max-min) ) + min;
#endif
}

void newSRAND()
{
    struct timespec c;
    clock_gettime(CLOCK_MONOTONIC, &c);
    srand(time(0)+c.tv_nsec);
}

/**********************************************/

void softmax_transform(float* w, const uint n)
{
    float d = 0;
    for(uint i = 0; i < n; i++)
        d += exp(w[i]);

    for(uint i = 0; i < n; i++)
        w[i] = exp(w[i]) / d;
}

static inline float softplus(const float x) //derivative is sigmoid()
{
    return log(1 + exp(x));
}

static inline float atanDerivative(const float x) //atan()
{
    return 1 / (1 + (x*x));
}

static inline float tanhDerivative(const float x) //tanh()
{
    return 1 - (x*x);
}

/**********************************************/

static inline float elu(const network* net, const float x)
{
    if(x <= 0){return -net->elualpha * (exp(x) - 1);}
    return x;
}

static inline float eluDerivative(const network* net, const float x)
{
    if(x > 0){return 1;}
    return net->elualpha * exp(x);
}

/**********************************************/

float gelu_table[] = {-0.0000000000000000002, -0.0000000000000000009, -0.0000000000000000036, -0.0000000000000000143, -0.0000000000000000548, -0.0000000000000002031, -0.0000000000000007294, -0.0000000000000024823, -0.0000000000000084470, -0.0000000000000276489, -0.0000000000000876395, -0.0000000000002696648, -0.0000000000008053627, -0.0000000000023360310, -0.0000000000065834627, -0.0000000000180345737, -0.0000000000480409849, -0.0000000001244949783, -0.0000000003139826100, -0.0000000007709958237, -0.0000000018440385905, -0.0000000042977204960, -0.0000000097641572436, -0.0000000216340466260, -0.0000000467654133713, -0.0000000986672599982, -0.0000002032635483159, -0.0000004090350635133, -0.0000008043631918483, -0.0000015463578046202, -0.0000029074279820216, -0.0000053483962825018, -0.0000096300309290754, -0.0000169783101176479, -0.0000293221871796576, -0.0000496255114760277, -0.0000823362420805513, -0.0001339751037145060, -0.0002138811629073111, -0.0003351222350318775, -0.0005155625743232390, -0.0007790584264589662, -0.0011567239437508328, -0.0016881768177933029, -0.0024226369449083331, -0.0034197156503302643, -0.0047497013495150517, -0.0064931242866868672, -0.0087393723504188264, -0.0115841355881902228, -0.0151254817281721715, -0.0194584105472432184, -0.0246678020996173135, -0.0308197626511069259, -0.0379514820900456115, -0.0460598465566349827, -0.0550891982744874525, -0.0649187979171901713, -0.0753507176218160962, -0.0860990653389843846, -0.0967815985533969636, -0.1069149061286552571, -0.1159143935198472808, -0.1231002669115984494, -0.1277105437559083734, -0.1289217945075437355, -0.1258778308708260185, -0.1177259095781044429, -0.1036592562322733446, -0.0829639004263334756, -0.0550670470622727504, -0.0195836003688628080, 0.0236428853258930649, 0.0745085457294865638, 0.1326301535941001586, 0.1973412045253939984, 0.2677045730992496098, 0.3425418649624665246, 0.4204782170639504435, 0.5000000000001483258, 0.5795217829363431550, 0.6574581350378183586, 0.7322954269010208961, 0.8026587954748576337, 0.8673698464061286861, 0.9254914542707162184, 0.9763571146742819895, 1.0195836003690090799, 1.0550670470623904063, 1.0829639004264233204, 1.1036592562323375155, 1.1177259095781453269, 1.1258778308708463634, 1.1289217945075467053, 1.1277105437558971879, 1.1231002669115759396, 1.1159143935198168052, 1.1069149061286192026, 1.0967815985533579948, 1.0860990653389441807, 1.0753507176217766972, 1.0649187979171523821, 1.0550891982744523556, 1.0460598465566031123, 1.0379514820900173078, 1.0308197626510822165, 1.0246678020995962921, 1.0194584105472255242, 1.0151254817281574194, 1.0115841355881782793, 1.0087393723504094467, 1.0064931242866794747, 1.0047497013495094720, 1.0034197156503259496, 1.0024226369449051877, 1.0016881768177909873, 1.0011567239437491050, 1.0007790584264577571, 1.0005155625743222814, 1.0003351222350314043, 1.0002138811629068815, 1.0001339751037141568, 1.0000823362420803164, 1.0000496255114759148, 1.0000293221871796590, 1.0000169783101175991, 1.0000096300309291308, 1.0000053483962825229, 1.0000029074279819241, 1.0000015463578046937, 1.0000008043631918309, 1.0000004090350635977, 1.0000002032635482152, 1.0000000986672599179, 1.0000000467654133196, 1.0000000216340467762, 1.0000000097641572605, 1.0000000042977204018, 1.0000000018440386995, 1.0000000007709957117, 1.0000000003139826177, 1.0000000001244950809, 1.0000000000480411266, 1.0000000000180346849, 1.0000000000065834005, 1.0000000000023361313, 1.0000000000008053558, 1.0000000000002695622, 1.0000000000000877076, 1.0000000000000277556, 1.0000000000000084377, 1.0000000000000024425, 1.0000000000000006661, 1.0000000000000002220, 1};

static inline float gelu(const float x)
{
    return 0.5 * x * (1 + tanh( 0.7978845608 * ( x + 0.044715 * (x*x*x) ) ));
}

static inline float geluDerivative(const float x)
{
    if(x <= 0){return -0.0000000000000000002;} else if (x >= 6.52){return 1;}
    const float fi = 0.133333333 * x;
    const uint i = 154.0 * fi;
    const float fx = x - floor(x);
    return gelu_table[i] * (1.f - fx) + gelu_table[i+1] * fx;
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

static inline float leaky_relu(const network* net, const float x)
{
    if(x < 0){return x * net->elualpha;}
    return x;
}

static inline float leaky_reluDerivative(const network* net, const float x)
{
    if(x > 0){return 1;}
    return net->elualpha;
}

/**********************************************/

static inline float relu(const float x)
{
    if(x < 0){return 0;}
    return x;
}

static inline float reluDerivative(const float x)
{
    if(x > 0){return 1;}
    return 0;
}

/**********************************************/

static inline float swish(const float x)
{
    return x / (1 + exp(-x));
}

static inline float swishDerivative(const float x)
{
    const float ex = exp(-x);
    const float oex = 1 + ex;
    return 1 + ex + x * ex / (oex * oex);
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

static inline float elliot_sigmoid(const float x) // aka softsign
{
    return x / (1 + fabs(x));
}

static inline float elliot_sigmoidDerivative(const float x)
{
    const float a = 1 - fabs(x);
    return a*a;
}

/**********************************************/

static inline float lecun_tanh(const float x)
{
    return 1.7159 * tanh(0.666666667 * x);
}

static inline float lecun_tanhDerivative(const float x)
{
    // this is a close enough approximation that
    // I literally "felt" the values for until
    // it seemed about as suitable as I could
    // get it to; 1.1439288854598999023
    // the maximum deviance of this function
    // against the proposed solution;
    // 1.14393 * pow(1 / cosh(x * 0.666667), 2)
    // is; 0.0000011390194490 at x = 0
    // Alternate from suggested by Wolfram:
    // -0.388522 * (-1.7159 + x) * (1.7159 + x);
    const float sx = x * 0.62331494;
    return 1.1439288854598999023-(sx*sx);
}

/**********************************************/

static inline float Derivative(const float x, const network* net)
{
    if(net->activator == 1)
        return atanDerivative(x);
    else if(net->activator == 2)
        return tanhDerivative(x);
    else if(net->activator == 3)
        return eluDerivative(net, x);
    else if(net->activator == 4)
        return leaky_reluDerivative(net, x);
    else if(net->activator == 5)
        return reluDerivative(x);
    else if(net->activator == 6)
        return sigmoidDerivative(x);
    else if(net->activator == 7)
        return swishDerivative(x);
    else if(net->activator == 8)
        return lecun_tanhDerivative(x);
    else if(net->activator == 9)
        return elliot_sigmoidDerivative(x);
    else if(net->activator == 10)
        return sigmoid(x); // this is the derivative of softplus
    else if(net->activator == 11)
        return geluDerivative(x);
    else if(net->activator == 12)
        return seluDerivative(x);
    
    return reluDerivative(x); // same as identity derivative
}

static inline float Activator(const float x, const network* net)
{
    if(net->activator == 1)
        return atan(x);
    else if(net->activator == 2)
        return tanh(x);
    else if(net->activator == 3)
        return elu(net, x);
    else if(net->activator == 4)
        return leaky_relu(net, x);
    else if(net->activator == 5)
        return relu(x);
    else if(net->activator == 6)
        return sigmoid(x);
    else if(net->activator == 7)
        return swish(x);
    else if(net->activator == 8)
        return lecun_tanh(x);
    else if(net->activator == 9)
        return elliot_sigmoid(x);
    else if(net->activator == 10)
        return softplus(x);
    else if(net->activator == 11)
        return gelu(x);
    else if(net->activator == 12)
        return selu(x);

    return x;
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
                return -1.75809934085;
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
                return -1.75809934085;
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
        return ERROR_ALLOC_PERCEPTRON_ALPHAWEIGHTS_FAIL;

    p->weights = weights;

    for(uint i = 0; i < p->weights; i++)
    {
        if(wit < 4)
            p->data[i] = qRandWeight(-d, d); // uniform
        else
            p->data[i] = qRandWeight(0, d);  // normal

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
            p->data[i] = qRandWeight(0, d);  // normal
        
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
}

void setWeightDropout(network* net, const float f)
{
    if(net == NULL){return;}
    net->wdropout = f;
}

void setDropoutDecay(network* net, const float f)
{
    if(net == NULL){return;}
    net->dropout_decay = f;
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

void setELUAlpha(network* net, const float f)
{
    if(net == NULL){return;}
    net->elualpha = f;
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
    net->activator  = uRand(0, 12);
    net->optimiser  = uRand(0, 4);
    net->rate       = uRandFloat(0.001, 0.1);
    net->dropout    = uRandFloat(0, 0.99);
    net->wdropout   = uRandFloat(0, 0.99);
    net->momentum   = uRandFloat(0.01, 0.99);
    net->rmsalpha   = uRandFloat(0.01, 0.99);
    net->elualpha   = uRandFloat(1e-4, 0.3);
    net->epsilon    = uRandFloat(1e-8, 1e-5);

    net->dropout_decay = uRandFloat(0, 0.99);
    if(net->dropout_decay < 0.1 || net->dropout_decay > 0.9)
        net->dropout_decay = 0;
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
        net->activator  = 2;
        net->optimiser  = 2;
        net->batches    = 3;
        net->rate       = 0.03;
        net->gain       = 1.0;
        net->dropout    = 0.3;
        net->wdropout   = 0;
     net->dropout_decay = 0;
        net->momentum   = 0.1;
        net->rmsalpha   = 0.2;
        net->elualpha   = 0.01;
        net->epsilon    = 1e-7;
        net->min_target = 0;
        net->max_target = 1;
    }
    net->cbatches   = 0;
    net->error      = 0;
    net->foutput    = 0;
    
    // create layers
    net->output = malloc((layers-1) * sizeof(float*));
    if(net->output == NULL)
        return ERROR_ALLOC_OUTPUT_ARRAY_FAIL;
    for(int i = 0; i < layers-1; i++)
    {
        net->output[i] = malloc(layers_size * sizeof(float));
        if(net->output[i] == NULL)
            return ERROR_ALLOC_OUTPUT_FAIL;
    }

    net->layer = malloc(layers * sizeof(ptron*));
    if(net->layer == NULL)
        return ERROR_ALLOC_LAYERS_ARRAY_FAIL;
    for(int i = 0; i < layers-1; i++)
    {
        net->layer[i] = malloc(layers_size * sizeof(ptron));
        if(net->layer[i] == NULL)
            return ERROR_ALLOC_LAYERS_FAIL;
    }

    net->layer[layers-1] = malloc(sizeof(ptron));
    if(net->layer[layers-1] == NULL)
        return ERROR_ALLOC_OUTPUTLAYER_FAIL;

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
        if(createPerceptron(&net->layer[0][i], inputs, d, net->init) < 0)
            return ERROR_CREATE_FIRSTLAYER_FAIL;
    
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
        for(int j = 0; j < layers_size; j++)
            if(createPerceptron(&net->layer[i][j], layers_size, d, net->init) < 0)
                return ERROR_CREATE_HIDDENLAYER_FAIL;

    // weight init
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(layers_size+1));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(layers_size+1));

    // create output layer
    if(createPerceptron(&net->layer[layers-1][0], layers_size, d, net->init) < 0)
        return ERROR_CREATE_OUTPUTLAYER_FAIL;

    // done
    return 0;
}

float processNetwork(network* net, float* inputs, const learn_type learn)
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

    // input layer
    for(int i = 0; i < net->num_layerunits; i++)
        of[0][i] = Activator(doPerceptron(inputs, &net->layer[0][i]), net);

    // hidden layers
    for(int i = 1; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            of[i][j] = Activator(doPerceptron(&of[i-1][0], &net->layer[i][j]), net);

    // binary classifier output layer
    const float output = sigmoid(doPerceptron(&of[net->num_layers-2][0], &net->layer[net->num_layers-1][0]));

    // if it's just forward pass, return result.
    if(learn == NO_LEARN)
        return output;

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

    // output derivative error layer before output layer
    float ler = 0;
    for(int j = 0; j < net->layer[net->num_layers-1][0].weights; j++)
        ler += net->layer[net->num_layers-1][0].data[j] * eout;
    ler += net->layer[net->num_layers-1][0].bias * eout;
    for(int i = 0; i < net->num_layerunits; i++)
        ef[net->num_layers-2][i] = net->gain * Derivative(of[net->num_layers-2][i], net) * ler;

    // output derivative error of all other layers
    for(int i = net->num_layers-3; i >= 0; i--)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            float ler = 0;
            for(int k = 0; k < net->layer[i+1][j].weights; k++)
                ler += net->layer[i+1][j].data[k] * ef[i+1][j];
            ler += net->layer[i+1][j].bias * ef[i+1][j];

            ef[i][j] = net->gain * Derivative(of[i][j], net) * ler;
        }
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
        
        if(fwrite(&net->dropout_decay, 1, sizeof(float), f) != sizeof(float))
            return -1;

        if(fwrite(&net->momentum, 1, sizeof(float), f) != sizeof(float))
            return -1;

        if(fwrite(&net->rmsalpha, 1, sizeof(float), f) != sizeof(float))
            return -1;

        if(fwrite(&net->elualpha, 1, sizeof(float), f) != sizeof(float))
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

    while(fread(&net->dropout_decay, 1, sizeof(float), f) != sizeof(float))
        return -1;

    while(fread(&net->momentum, 1, sizeof(float), f) != sizeof(float))
        return -1;

    while(fread(&net->rmsalpha, 1, sizeof(float), f) != sizeof(float))
        return -1;

    while(fread(&net->elualpha, 1, sizeof(float), f) != sizeof(float))
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
