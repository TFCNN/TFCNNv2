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
    SELU        = 12, // not sure, implementation seems correct, derivative & activation are ok, alpha dropout seems ok ?
    BENT        = 13,
    GAUSSIAN    = 14, // not w.r.t f(x), broken
    SINUSOID    = 15,
    SINC        = 16, // not w.r.t f(x), broken
    ISRU        = 17, // not w.r.t f(x), broken
    SQNL        = 18  // not w.r.t f(x), broken
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
void setDropoutDecay(network* net, const float f); //Set dropout to silence the unit activation by decay rather than on/off
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
    double u = ((float)rand() / (float)RAND_MAX) * 2 - 1;
    double v = ((float)rand() / (float)RAND_MAX) * 2 - 1;
    double r = u * u + v * v;
    while(r == 0 || r > 1)
    {
        u = ((float)rand() / (float)RAND_MAX) * 2 - 1;
        v = ((float)rand() / (float)RAND_MAX) * 2 - 1;
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
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    double u = ((float)rand() / (float)RAND_MAX) * 2 - 1;
    double v = ((float)rand() / (float)RAND_MAX) * 2 - 1;
    double r = u * u + v * v;
    while(r == 0 || r > 1)
    {
        u = ((float)rand() / (float)RAND_MAX) * 2 - 1;
        v = ((float)rand() / (float)RAND_MAX) * 2 - 1;
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

// well, it's never going to be used, but I left it in here anyway.
void softmax_transform(float* w, const uint n)
{
    float d = 0;
    for(uint i = 0; i < n; i++)
        d += exp(w[i]);

    for(uint i = 0; i < n; i++)
        w[i] = exp(w[i]) / d;
}

// I would like to eventually compact the lookup code into a single swiss-army like function, this function is the working towards that.
float table_lerp(const float sa, const float ia, const float sb, const float ib, const float i)
{
    return sa + (( (i - ia) / (ib - ia) ) * (sb - sa));
}
float table_derivative(const float* ts, const float* ti, const uint table_size, const float fi, const float fn)
{
    const uint i = (uint)(table_size * fn);
    return table_lerp(ts[i], ti[i], ts[i+1], ti[i+1], fi);
}
// don't laugh at me please, i am serious. :'( you have better way ? you email me ^ in header
float find_derivative(const float* ts, const float* ti, const uint table_size, const float fi)
{
    for(uint i = 0; i < table_size; i++)
    {
        if(fi > 0)
        {
            if(fi >= ti[i] && fi < ti[i+1])
                return table_lerp(ts[i], ti[i], ts[i+1], ti[i+1], fi);
        }
        else
        {
            if(fi <= ti[i] && fi > ti[i+1])
                return table_lerp(ts[i], ti[i], ts[i+1], ti[i+1], fi);
        }
    }
    return table_derivative(ts, ti, table_size, fi, (fi+7.5)*0.066666667);
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

static inline float gelu(const float x)
{
    return 0.5 * x * (1 + tanh( 0.7978845608 * ( x + 0.044715 * (x*x*x) ) ));
}

// this is really not good
const float gelu_derivative_table_sample[] = {-0.0000000000000000548, -0.0000000000000002031, -0.0000000000000007294, -0.0000000000000024823, -0.0000000000000084470, -0.0000000000000276489, -0.0000000000000876395, -0.0000000000002696648, -0.0000000000008053627, -0.0000000000023360310, -0.0000000000065834627, -0.0000000000180345737, -0.0000000000480409849, -0.0000000001244949783, -0.0000000003139826100, -0.0000000007709958237, -0.0000000018440385905, -0.0000000042977204960, -0.0000000097641572436, -0.0000000216340466259, -0.0000000467654133712, -0.0000000986672599981, -0.0000002032635483156, -0.0000004090350635127, -0.0000008043631918472, -0.0000015463578046182, -0.0000029074279820179, -0.0000053483962824953, -0.0000096300309290642, -0.0000169783101176286, -0.0000293221871796255, -0.0000496255114759754, -0.0000823362420804674, -0.0001339751037143743, -0.0002138811629071090, -0.0003351222350315721, -0.0005155625743227850, -0.0007790584264583070, -0.0011567239437499457, -0.0016881768177920927, -0.0024226369449066249, -0.0034197156503280191, -0.0047497013495120966, -0.0064931242866830049, -0.0087393723504138720, -0.0115841355881839882, -0.0151254817281644624, -0.0194584105472338717, -0.0246678020996062321, -0.0308197626510939224, -0.0379514820900307207, -0.0460598465566181350, -0.0550891982744689257, -0.0649187979171702290, -0.0753507176217952240, -0.0860990653389632071, -0.0967815985533763967, -0.1069149061286362723, -0.1159143935198312381, -0.1231002669115865700, -0.1277105437559023227, -0.1289217945075453453, -0.1258778308708368154, -0.1177259095781261478, -0.1036592562323071509, -0.0829639004263808544, -0.0550670470623347286, -0.0195836003689398852, 0.0236428853258009164, 0.0745085457293799269, 0.1326301535939796161, 0.1973412045252614933, 0.2677045730991071126, 0.3425418649623165335, 0.4204782170637959005, 0.4999999999999921174, 0.5795217829361885009, 0.6574581350376682565, 0.7322954269008783434, 0.8026587954747251841, 0.8673698464060081159, 0.9254914542706094149, 0.9763571146741896190, 1.0195836003689320304, 1.0550670470623284558, 1.0829639004263762470, 1.1036592562323037647, 1.1177259095781240106, 1.1258778308708357052, 1.1289217945075451510, 1.1277105437559029610, 1.1231002669115879300, 1.1159143935198327924, 1.1069149061286380764, 1.0967815985533784229, 1.0860990653389654970, 1.0753507176217973473, 1.0649187979171723661, 1.0550891982744710074, 1.0460598465566199877, 1.0379514820900321848, 1.0308197626510950951, 1.0246678020996073943, 1.0194584105472348501, 1.0151254817281651910, 1.0115841355881844965, 1.0087393723504143317, 1.0064931242866834715, 1.0047497013495123586, 1.0034197156503281700, 1.0024226369449067420, 1.0016881768177923195, 1.0011567239437499932, 1.0007790584264584233, 1.0005155625743227255, 1.0003351222350316263, 1.0002138811629071036, 1.0001339751037143788, 1.0000823362420805385, 1.0000496255114759148, 1.0000293221871796590, 1.0000169783101175991, 1.0000096300309291308, 1.0000053483962825229, 1.0000029074279819241, 1.0000015463578046937, 1.0000008043631918309, 1.0000004090350635977, 1.0000002032635482152, 1.0000000986672599179, 1.0000000467654133196, 1.0000000216340467762, 1.0000000097641572605, 1.0000000042977204018, 1.0000000018440386995, 1.0000000007709957117, 1.0000000003139826177, 1.0000000001244950809, 1.0000000000480411266, 1.0000000000180346849, 1.0000000000065834005, 1.0000000000023361313, 1.0000000000008053558, 1.0000000000002695622, 1.0000000000000877076, 1.0000000000000277556, 1.0000000000000084377, 1.0000000000000024425, 1.0000000000000006661, 1.0000000000000002220, 1.0000000000000000000};
const float gelu_derivative_table_input[] = {-0.0000000000000000000, -0.0000000000000000000, -0.0000000000000000000, -0.0000000000000003997, -0.0000000000000007883, -0.0000000000000023315, -0.0000000000000076605, -0.0000000000000237810, -0.0000000000000728972, -0.0000000000002168932, -0.0000000000006278311, -0.0000000000017667646, -0.0000000000048362815, -0.0000000000128836273, -0.0000000000334124568, -0.0000000000843964898, -0.0000000002077121530, -0.0000000004983163326, -0.0000000011658348731, -0.0000000026609368042, -0.0000000059276401565, -0.0000000128931381127, -0.0000000273934528394, -0.0000000568757023700, -0.0000001154453528329, -0.0000002291796192821, -0.0000004451490553947, -0.0000008463364338240, -0.0000015756812672407, -0.0000028738027140207, -0.0000051367373998801, -0.0000090019366325578, -0.0000154731515067397, -0.0000260970919043757, -0.0000432063970947638, -0.0000702459510648623, -0.0001121976456488483, -0.0001761192543199286, -0.0002718059404287487, -0.0004125804407522082, -0.0006161976489238441, -0.0009058508439920843, -0.0013112284941598773, -0.0018695648759603500, -0.0026266109198331833, -0.0036373920738697052, -0.0049666641280055046, -0.0066888942383229733, -0.0088875927031040192, -0.0116539206355810165, -0.0150842657312750816, -0.0192768424749374390, -0.0243270322680473328, -0.0303214695304632187, -0.0373310036957263947, -0.0454023070633411407, -0.0545487664639949799, -0.0647404789924621582, -0.0758941024541854858, -0.0878630355000495911, -0.1004284247756004333, -0.1132919341325759888, -0.1260709911584854126, -0.1382972300052642822, -0.1494189500808715820, -0.1588080078363418579, -0.1657715141773223877, -0.1695683151483535767, -0.1694298684597015381, -0.1645848006010055542, -0.1542859971523284912, -0.1378388255834579468, -0.1146290823817253113, -0.0841485708951950073, -0.0460172481834888458, -0.0000000000000049544, 0.0539827533066272736, 0.1158514320850372314, 0.1853709369897842407, 0.2621611654758453369, 0.3457140028476715088, 0.4354152083396911621, 0.5305701494216918945, 0.6304317116737365723, 0.7342284917831420898, 0.8411920070648193359, 0.9505810737609863281, 1.0617028474807739258, 1.1739289760589599609, 1.2867079973220825195, 1.3995715379714965820, 1.5121369361877441406, 1.6241059303283691406, 1.7352594137191772461, 1.8454512357711791992, 1.9545977115631103516, 2.0626688003540039062, 2.1696786880493164062, 2.2756729125976562500, 2.3807232379913330078, 2.4849157333374023438, 2.5883460044860839844, 2.6911125183105468750, 2.7933111190795898438, 2.8950333595275878906, 2.9963626861572265625, 3.0973732471466064453, 3.1981303691864013672, 3.2986886501312255859, 3.3990943431854248047, 3.4993836879730224609, 3.5995874404907226562, 3.6997282505035400391, 3.7998237609863281250, 3.8998878002166748047, 3.9999296665191650391, 4.0999565124511718750, 4.1999735832214355469, 4.2999849319458007812, 4.3999910354614257812, 4.4999947547912597656, 4.5999970436096191406, 4.6999983787536621094, 4.7999992370605468750, 4.8999996185302734375, 5.0000000000000000000, 5.0999999046325683594, 5.1999998092651367188, 5.3000001907348632812, 5.4000000953674316406, 5.5000000000000000000, 5.5999999046325683594, 5.6999998092651367188, 5.8000001907348632812, 5.9000000953674316406, 6.0000000000000000000, 6.0999999046325683594, 6.1999998092651367188, 6.3000001907348632812, 6.4000000953674316406, 6.5000000000000000000, 6.5999999046325683594, 6.6999998092651367188, 6.8000001907348632812, 6.9000000953674316406, 7.0000000000000000000, 7.0999999046325683594, 7.1999998092651367188, 7.3000001907348632812, 7.4000000953674316406, 7.5000000000000000000};
static inline float geluDerivative(const float x)
{
    return find_derivative(&gelu_derivative_table_sample[0], &gelu_derivative_table_input[0], 150, x);
}

/**********************************************/

static inline float selu(const float x)
{
    if(x < 0){return 1.0507 * (1.67326 * exp(x) - 1.67326);}
    return 1.0507 * x;
}

// this is a good enough approximation
const float selu_derivative_table[] = {0, 0.0109884506091475487, 0.0219765231013298035, 0.0329649448394775391, 0.0439531728625297546, 0.0549414157867431641, 0.0659300759434700012, 0.0769182965159416199, 0.0879065990447998047, 0.0988955870270729065, 0.1098842471837997437, 0.1208725571632385254, 0.1318618506193161011, 0.1428511738777160645, 0.1538401097059249878, 0.1648288071155548096, 0.1758172214031219482, 0.1868053525686264038, 0.1977937221527099609, 0.2087835371494293213, 0.2197735160589218140, 0.2307635843753814697, 0.2417530715465545654, 0.2527414858341217041, 0.2637297213077545166, 0.2747193872928619385, 0.2857088744640350342, 0.2966979444026947021, 0.3076872825622558594, 0.3186781108379364014, 0.3296684622764587402, 0.3406589925289154053, 0.3516495227813720703, 0.3626385927200317383, 0.3736267387866973877, 0.3846164345741271973, 0.3956083655357360840, 0.4065967500209808350, 0.4175849854946136475, 0.4285739660263061523, 0.4395658969879150391, 0.4505554437637329102, 0.4615469872951507568, 0.4725368618965148926, 0.4835268855094909668, 0.4945199191570281982, 0.5055096745491027832, 0.5165005922317504883, 0.5274926424026489258, 0.5384813547134399414, 0.5494734048843383789, 0.5604652166366577148, 0.5714537501335144043, 0.5824418663978576660, 0.5934332013130187988, 0.6044262051582336426, 0.6154194474220275879, 0.6264118552207946777, 0.6374027132987976074, 0.6483916044235229492, 0.6593850255012512207, 0.6703768968582153320, 0.6813676953315734863, 0.6923584342002868652, 0.7033503651618957520, 0.7143450975418090820, 0.7253373861312866211, 0.7363291978836059570, 0.7473229169845581055, 0.7583137154579162598, 0.7693043947219848633, 0.7802980542182922363, 0.7912903428077697754, 0.8022847771644592285, 0.8132773041725158691, 0.8242717981338500977, 0.8352643847465515137, 0.8462594747543334961, 0.8572533726692199707, 0.8682422637939453125, 0.8792311549186706543, 0.8902254104614257812, 0.9012217521667480469, 0.9122169017791748047, 0.9232075214385986328, 0.9341995716094970703, 0.9451900124549865723, 0.9561852216720581055, 0.9671824574470520020, 0.9781787991523742676, 0.9891714453697204590, 1.0001673698425292969, 1.0111640691757202148, 1.0221587419509887695, 1.0331488847732543945, 1.0441422462463378906, 1.0551362037658691406, 1.0661286115646362305, 1.0771168470382690430, 1.0881093740463256836, 1.0991039276123046875, 1.1100984811782836914, 1.1210907697677612305, 1.1320898532867431641, 1.1430823802947998047, 1.1540776491165161133, 1.1650736331939697266, 1.1760684251785278320, 1.1870599985122680664, 1.1980583667755126953, 1.2090495824813842773, 1.2200438976287841797, 1.2310396432876586914, 1.2420349121093750000, 1.2530280351638793945, 1.2640172243118286133, 1.2750133275985717773, 1.2860021591186523438, 1.2969946861267089844, 1.3079894781112670898, 1.3189851045608520508, 1.3299798965454101562, 1.3409724235534667969, 1.3519608974456787109, 1.3629575967788696289, 1.3739541769027709961, 1.3849468231201171875, 1.3959437608718872070, 1.4069435596466064453, 1.4179450273513793945, 1.4289467334747314453, 1.4399474859237670898, 1.4509457349777221680, 1.4619404077529907227, 1.4729300737380981445, 1.4839280843734741211, 1.4949184656143188477, 1.5059150457382202148, 1.5169166326522827148, 1.5279068946838378906, 1.5388998985290527344, 1.5498945713043212891, 1.5608897209167480469, 1.5718842744827270508, 1.5828770399093627930, 1.5938670635223388672, 1.6048692464828491211, 1.6158665418624877930, 1.6268578767776489258, 1.6378585100173950195, 1.6488510370254516602, 1.6598509550094604492, 1.6708408594131469727, 1.6818361282348632812, 1.6928360462188720703, 1.7038397789001464844, 1.7148313522338867188, 1.7258248329162597656, 1.7368186712265014648, 1.7478123903274536133};
static inline float seluDerivative(const float x)
{
    if(x <= -1.758094){return 0.0000001978474415409;} else if (x >= -0.0167265){return 1.0507;}
    const uint i = (uint)(160.0 * (1-(0.5687978 * -x)));
    const float fx = ((1.758094-(-x)) - selu_derivative_table[i]) * 91.007652196;
    return selu_derivative_table[i] * (1.f - fx) + selu_derivative_table[i+1] * fx;
}

/**********************************************/

static inline float bent(const float x)
{
    return ((sqrt(x*x + 1)-1) * 0.5) + x;
}

static inline float bentDerivative(const float x)
{
    return (x / (2 * sqrt(x*x + 1))) + 1;
}

/**********************************************/

static inline float gauss(const float x)
{
    if(x < -6 || x > 6){return 4311231531843584;}
    return exp(-x*-x);
}

static inline float gaussDerivative(const float x)
{
    if(x < -6 || x > 6){return 4311231531843584;}
    return -2 * x * exp(-x*-x);
}

/**********************************************/

static inline float sinusoid(const float x)
{
    return sin(x);
}

// it's pretty good, just tapers off at 0 faster than the original
const float sinusoid_derivative_table[] = {0, 0.1581410318613052368, 0.2229219228029251099, 0.2723294794559478760, 0.3134416639804840088, 0.3492934703826904297, 0.3818698227405548096, 0.4112455546855926514, 0.4384033977985382080, 0.4633952379226684570, 0.4871506094932556152, 0.5096907019615173340, 0.5310391187667846680, 0.5512208342552185059, 0.5702633857727050781, 0.5881958603858947754, 0.6050477623939514160, 0.6216329336166381836, 0.6371735334396362305, 0.6524592638015747070, 0.6667391061782836914, 0.6807782649993896484, 0.6945716738700866699, 0.7074079513549804688, 0.7200149893760681152, 0.7323887348175048828, 0.7438573241233825684, 0.7551108598709106445, 0.7661462426185607910, 0.7769601941108703613, 0.7869330048561096191, 0.7967043519020080566, 0.8062717318534851074, 0.8156327605247497559, 0.8247849345207214355, 0.8337260484695434570, 0.8419144749641418457, 0.8499135375022888184, 0.8577213287353515625, 0.8653361797332763672, 0.8727563619613647461, 0.8799801468849182129, 0.8870059251785278320, 0.8938322067260742188, 0.9000219106674194336, 0.9060353040695190430, 0.9118710756301879883, 0.9175280928611755371, 0.9230053424835205078, 0.9283016324043273926, 0.9334160685539245605, 0.9383474588394165039, 0.9430950284004211426, 0.9476577043533325195, 0.9520346522331237793, 0.9562250375747680664, 0.9602280259132385254, 0.9637765288352966309, 0.9671621918678283691, 0.9703844785690307617, 0.9734427332878112793, 0.9763364791870117188, 0.9790652394294738770, 0.9816285967826843262, 0.9840259552001953125, 0.9862570762634277344, 0.9883215427398681641, 0.9902189373970031738, 0.9919490218162536621, 0.9935114383697509766, 0.9949059486389160156, 0.9961323738098144531, 0.9971904158592224121, 0.9980799555778503418, 0.9988007545471191406, 0.9993528127670288086, 0.9997360110282897949, 0.9999502301216125488, 0.9999954104423522949, 0.9998716711997985840, 0.9995788931846618652, 0.9991172552108764648, 0.9984866976737976074, 0.9976874589920043945, 0.9967195987701416016, 0.9955832362174987793, 0.9942786693572998047, 0.9928060770034790039, 0.9911656975746154785, 0.9893578290939331055, 0.9873827099800109863, 0.9852407574653625488, 0.9829323291778564453, 0.9804577231407165527, 0.9778174757957458496, 0.9750119447708129883, 0.9720416665077209473, 0.9689070582389831543, 0.9656088352203369141, 0.9621473550796508789, 0.9582378268241882324, 0.9541405439376831055, 0.9498561620712280273, 0.9453856945037841797, 0.9407299160957336426, 0.9358897209167480469, 0.9308661222457885742, 0.9256600737571716309, 0.9202726483345031738, 0.9147047996520996094, 0.9089577198028564453, 0.9030324816703796387, 0.8969302177429199219, 0.8906522393226623535, 0.8837320804595947266, 0.8766131401062011719, 0.8692969679832458496, 0.8617851734161376953, 0.8540794849395751953, 0.8461816906929016113, 0.8380934596061706543, 0.8298166990280151367, 0.8207824230194091797, 0.8115380406379699707, 0.8020858764648437500, 0.7924283742904663086, 0.7825680971145629883, 0.7718720436096191406, 0.7609529495239257812, 0.7498139739036560059, 0.7384582757949829102, 0.7262020707130432129, 0.7137106060981750488, 0.7009878754615783691, 0.6880380511283874512, 0.6741271018981933594, 0.6599727272987365723, 0.6448161602020263672, 0.6294016838073730469, 0.6129456162452697754, 0.5962193012237548828, 0.5784145593643188477, 0.5603299140930175781, 0.5411334633827209473, 0.5207961797714233398, 0.4992926120758056641, 0.4765980839729309082, 0.4526898562908172607, 0.4266435205936431885, 0.3984046876430511475, 0.3679239153861999512, 0.3342163562774658203, 0.2962574362754821777, 0.2520225048065185547, 0.1974607855081558228, 0.1194772347807884216, 0};
static inline float sinusoidDerivative(const float x)
{
    if (x <= -1) {return 0;} else if (x >= 1) {return 0;}
    const float fn = (x+1)*0.5;
    const float fi = 156.0 * fn;
    const uint  i  = (uint)fi;
    const float fd = fi - floor(fi);
    return sinusoid_derivative_table[i] * (1.f - fd) + sinusoid_derivative_table[i+1] * fd;
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

static inline float isru(const float x)
{
    return x / sqrt(1+x*x);
}

static inline float isruDerivative(const float x)
{
    return pow((1 / sqrt(1 + x*x)), 3);
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
    // Alternate form as suggested by Wolfram:
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
    else if(net->activator == 13)
        return bentDerivative(x);
    else if(net->activator == 14)
        return gaussDerivative(x);
    else if(net->activator == 15)
        return sinusoidDerivative(x);
    else if(net->activator == 16)
        return sincDerivative(x);
    else if(net->activator == 17)
        return isruDerivative(x);
    else if(net->activator == 18)
        return sqnlDerivative(x);
    
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
    else if(net->activator == 13)
        return bent(x);
    else if(net->activator == 14)
        return gauss(x);
    else if(net->activator == 15)
        return sinusoid(x);
    else if(net->activator == 16)
        return sinc(x);
    else if(net->activator == 17)
        return isru(x);
    else if(net->activator == 18)
        return sqnl(x);

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
            {
                return f * net->drop_a + net->drop_b;
            }
            else
            {
                if(net->dropout_decay != 0)
                    return f * (1.0 - net->dropout_decay);
                else
                    return 0;
            }
        }
    }
    else if(type == 1)
    {
        if(net->wdropout == 0)
            return f;

        if(uRandFloat(0, 1) <= net->wdropout)
        {
            if(net->activator == SELU)
            {
                return f * net->drop_wa + net->drop_wb;
            }
            else
            {
                if(net->dropout_decay != 0)
                    return f * (1.0 - net->dropout_decay);
                else
                    return 0;
            }
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
            p->data[i] = qRandWeight(-1, 1) * d; // uniform
        else
            p->data[i] = qRandNormal() * d; // normal

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
            p->data[i] = qRandWeight(-1, 1) * d; // uniform
        else
            p->data[i] = qRandNormal() * d; // normal
        
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
    net->activator  = uRand(0, 18);
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

    float d1 = 1 - net->dropout;
    net->drop_a = pow(net->dropout + 3.090895504 * net->dropout * d1, -0.5);
    net->drop_b = -net->drop_a * (d1 * -1.758094282);
    
    d1 = 1 - net->wdropout;
    net->drop_wa = pow(net->wdropout + 3.090895504 * net->wdropout * d1, -0.5);
    net->drop_wb = -net->drop_wa * (d1 * -1.758094282);
    
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
