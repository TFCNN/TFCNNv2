## TFCNNv2 - Tiny Fully Connected Neural Network Library [WIP]
_`TFCNNv2` and all variants are targeted at Linux / Unix / BSD platforms; for a truly cross-platform implementation please refer to [`TFCNNv1`](https://github.com/TFCNN/TFCNNv1)._

`[9/11/20]` - Added [`TFCNNv21.h`](https://github.com/TFCNN/TFCNNv2/blob/main/TFCNNV2.1) for more accurate computations of `SELU, GELU, MISH, SQNL` and `SINC`.
<br>`[11/11/20]` - Added [`TFCNNv2_softmax.h`](https://github.com/TFCNN/TFCNNv2/blob/main/TFCNNv2_softmax.h) for multiple classification using a softmax layer with cross-entropy loss.
<br>`[12/11/20]` - Added [`TFCNNv2_multiclass.h`](https://github.com/TFCNN/TFCNNv2/blob/main/TFCNNv2_multiclass.h) for regular multiple classification on the same network, absolute loss.
<br>`[01/01/21]` - Added optional loss functions in [`TFCNNv2.h`](https://github.com/TFCNN/TFCNNv2/blob/main/TFCNNv2.h) and corrected the code to allow non-destructive use of changing the uint defitionion to smaller storage types.

**Notice:** you cannot share `saveNetwork()` files between versions, each version saves a different format of the `saveNetwork()` file.

### Version 2
```
// primary function set
int   createNetwork(network* net, const weight_init_type init_type, const uint num_inputs, const uint num_hidden_layers, const uint num_layer_units, const uint default_settings);
float processNetwork(network* net, const float* inputs, const learn_type learn);
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
void setLoss(network* net, const loss u);
void setUnitDropout(network* net, const float f); //Dropout
void setWeightDropout(network* net, const float f); //Drop Connect
void setDropoutDecay(network* net, const float f); //Set dropout to silence the unit activation by decay rather than on/off
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
    WEIGHT_INIT_UNIFORM_RELU        = 4, // he initialisation
    WEIGHT_INIT_NORMAL              = 5,
    WEIGHT_INIT_NORMAL_GLOROT       = 6,
    WEIGHT_INIT_NORMAL_LECUN        = 7,
    WEIGHT_INIT_NORMAL_LECUN_POW    = 8,
    WEIGHT_INIT_NORMAL_RELU         = 9  // he initialisation
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
    GELU        = 11, // w.r.t derivative; lookup-table, 0.1 to 0.5 has a chunk missing with an avg deviance of 0.25; out of a 0-1 total range; it sounds like a lot but only makes up 5.38% of the total distribution
    SELU        = 12, // w.r.t derivative; lookup-table, not sure, implementation seems correct, derivative & activation are ok, alpha dropout seems ok ?
    BENT        = 13,
    BISIGMOID   = 14, // w.r.t derivative; lookup-table
    SINUSOID    = 15, // w.r.t derivative; lookup-table
    SINC        = 16, // w.r.t derivative; It's w.r.t x and not f(x), not ideal, but not that off either
    ISRU        = 17, // w.r.t derivative; lookup-table
    SQNL        = 18, // w.r.t derivative; lookup-table, not perfect, but not that off, ~0.30 for 30% of the total distribution
    MISH        = 19  // w.r.t derivative; lookup-table, same problem as GELU
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

enum 
{
    LOSS_ABSOLUTE       = 0,
    LOSS_SQUARED        = 1,
    LOSS_MEANSQUARED    = 2,
    LOSS_CROSSENTROPY   = 3
}
typedef loss;
```
