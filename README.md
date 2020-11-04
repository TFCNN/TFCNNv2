## TFCNN - Tiny Fully Connected Neural Network Library

### Version 2

- `TFCNNv2.h` is a Linux / Unix / BSD version.
```
// primary function set
int   createNetwork(network* net, const weight_init_type init_type, const uint num_inputs, const uint num_hidden_layers, const uint num_layer_units, const uint default_settings);
float processNetwork(network* net, float* inputs, const learn_type learn);
void  resetNetwork(network* net);
void  destroyNetwork(network* net);
int   saveNetwork(network* net, const char* file);
int   loadNetwork(network* net, const char* file);

// accessors
void setWeightInit(network* net, const weight_init_type u);
void setOptimiser(network* net, const optimiser u);
void setActivator(network* net, const activator u);
void setBatches(network* net, const uint u);
void setLearningRate(network* net, const float f);
void setGain(network* net, const float f);
void setUnitDropout(network* net, const float f); //Dropout
void setWeightDropout(network* net, const float f); //Drop Connect
void setDropoutDecay(network* net, const float f); //Set sropout to silence the unit activation by decay rather than on/off
void setMomentum(network* net, const float f); //SGDM & NAG
void setRMSAlpha(network* net, const float f);
void setELUAlpha(network* net, const float f); //ELU & LeakyReLU
void setEpsilon(network* net, const float f); //ADA & RMS
void setTargetMin(network* net, const float f);
void setTargetMax(network* net, const float f);
void randomHyperparameters(network* net);

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
```

#### ENUMS
```
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
    ELLIOT      = 9, // aka softsign
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
```
