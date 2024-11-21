#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define ld long double

// https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
const int MAXN = 6e4 + 7;
const int TESTN = 10000;
unsigned int image[MAXN][30][30];
unsigned int test[TESTN][30][30];
unsigned int num, magic, rows, cols;
unsigned int image_label[MAXN];
unsigned int test_label[TESTN];

unsigned int in(ifstream& icin, unsigned int size) {
    unsigned int ans = 0;
    for (int i = 0; i < size; i++) {
        unsigned char x;
        icin.read((char*)&x, 1);
        unsigned int temp = x;
        ans <<= 8;
        ans += temp;
    }
    return ans;
}
void input() {
    ifstream icin;
    icin.open("./assets/train-images.idx3-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
    for (int i = 0; i < num; i++) {
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < cols; y++) {
                image[i][x][y] = in(icin, 1);
            }
        }
    }
    icin.close();
    icin.open("./assets/train-labels.idx1-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4);
    for (int i = 0; i < num; i++) {
        image_label[i] = in(icin, 1);
    }
    icin.close();

    icin.open("./assets/t10k-images.idx3-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
    for (int i = 0; i < num; i++) {
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < cols; y++) {
                test[i][x][y] = in(icin, 1);
            }
        }
    }
    icin.close();
    icin.open("./assets/t10k-labels.idx1-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4);
    for (int i = 0; i < num; i++) {
        test_label[i] = in(icin, 1);
    }
    icin.close();
}


vector<vector<ld>> mat_mult(vector<vector<ld>> A, vector<vector<ld>> B, bool comment = false){
    int n = A.size();
    int k = B.size();
    int m = B[0].size();
    // A = n rows, k elements;
    // B = k rows, m elements;
    // C = n rows, m elements;
    vector<vector<ld>> C (n, vector<ld> (m));

    if(comment){
        cout << "A: " << A.size() << " " << A[0].size() << "\n";
        cout << "B: " << B.size() << " " << B[0].size() << "\n";
        cout << "C: " << C.size() << " " << C[0].size() << "\n";
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            // index i,j in C, determined by:
                //sum of (i,k in A) * (k, j in B)
            C[i][j] = 0;
            for(int x = 0; x < k; x++){
                C[i][j] += (ld)A[i][x] * B[x][j];
            }
        }
    }

    return C;
}

vector<ld> pair_mult(vector<ld> A, vector<ld> B){
    int n = A.size();
    vector<ld> r (n);

    for(int i = 0; i < n; i++){
        r[i] = A[i] * B[i];
    }

    return r;
}

/*
ADD 2D VECTORS
*/
vector<vector<ld>> mat_add(vector<vector<ld>> A, vector<vector<ld>> B){
    int n = A.size();
    int m = A[0].size();
    vector<vector<ld>> R (n, vector<ld> (m));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            R[i][j] = A[i][j] + B[i][j];
        }
    }

    return R;
}

/*
SUB 2D VECTORS
*/
vector<vector<ld>> mat_sub(vector<vector<ld>> A, vector<vector<ld>> B){
    int n = A.size();
    int m = A[0].size();
    vector<vector<ld>> R (n, vector<ld> (m));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            R[i][j] = A[i][j] - B[i][j];
        }
    }

    return R;
}

/*
TRANSPOSE A
*/
vector<vector<ld>> transpose(vector<vector<ld>> A){
    int n = A.size();
    int m = A[0].size();
    vector<vector<ld>> R (m, vector<ld> (n));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            R[j][i] = A[i][j];
        }
    }

    return R;
}

/*
MULTIPLY VECTOR BY A SCALAR
*/
vector<ld> scalar_mult(ld c, vector<ld> A){
    int n = A.size();
    vector<ld> R (n);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < 1; j++){
            R[i] = c * A[i];
        }
    }

    return R;
}


/*
MAKE VECTOR INTO 2D ARRAY WITH VALUES IN THE COLUMN
*/
vector<vector<ld>> vec_to_col(vector<ld> A){
    int n = A.size();
    vector<vector<ld>> R (1, vector<ld> (n));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < 1; j++){
            R[j][i] = A[i];
        }
    }

    return R;
}


/*
MAKE VECTOR INTO 2D ARRAY WITH VALUES IN THE ROW
*/
vector<vector<ld>> vec_to_row(vector<ld> A){
    int n = A.size();
    vector<vector<ld>> R (n, vector<ld> (1));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < 1; j++){
            R[i][j] = A[i];
        }
    }

    return R;
}

void print_size(vector<ld> &A){
    cout << "size: " << A.size() << "\n";
}

void print_size(vector<vector<ld>> &A){
    cout << "size: " << A.size() << " x " << A[0].size() << "\n";
}

/*
TESTS:
    {
    lr = 0.0003
    hidden_count = 128
    epochs = 5
    }
    => 51% correct

    {
    lr = 0.0003;
    hidden_count = 64;
    middle_count = 32;
    epochs = 5
    }
    => 59% correct
    (TESTED: 5000/5000 -> 0.4122)



*/

struct ML {
    ld lr = 0.0003;
    int input_count = 28*28;
    int hidden_count = 64;
    int middle_count = 32;

    int output_count = 10;

    int guess = 0;
    
    // [INPUT_COUNT]
    vector<ld> input = vector<ld> (28*28);

    // [OUTPUT_COUNT]
    vector<ld> label = vector<ld> (10,0);

    // [INPUT_COUNT][HIDDEN_COUNT]
    vector<vector<ld>> wih;

    // [HIDDEN_COUNT][HIDDEN_COUNT]
    vector<vector<ld>> whm;

    // [HIDDEN_COUNT][OUTPUT_COUNT]
    vector<vector<ld>> wmo;

    // [HIDDEN_COUNT]
    vector<ld> bih;

    // [HIDDEN_COUNT]
    vector<ld> bhm;

    // [OUTPUT_COUNT]
    vector<ld> bmo;

    // [LAYERS][BATCH_SIZE][INPUT_COUNT]
    vector<vector<vector<ld>>> activations {};

    // [BATCH_SIZE][OUTPUT_COUNT]
    vector<vector<ld>> one_hot_out;

    ML() {
        default_random_engine generator;
        normal_distribution<long double> hidden_distribution(0, 1 / sqrt(input_count));
        normal_distribution<long double> middle_distribution(0, 1 / sqrt(hidden_count));
        normal_distribution<long double> output_distribution(0, 1 / sqrt(middle_count));

        // SET HIDDEN WEIGHTS TO STANDARD DEVIATION OF sqrt of input_count
        wih.resize(input_count, vector<ld> (hidden_count));
        bih.resize(hidden_count);
        for(int i = 0; i < input_count; i++){
            for(int j = 0; j < hidden_count; j++){
                wih[i][j] = hidden_distribution(generator);
            }
        }

        // SET HIDDEN WEIGHTS TO STANDARD DEVIATION OF sqrt of hidden_count
        whm.resize(hidden_count, vector<ld> (middle_count));
        bhm.resize(middle_count);
        for(int i = 0; i < hidden_count; i++){
            for(int j = 0; j < middle_count; j++){
                whm[i][j] = middle_distribution(generator);
            }
        }

        // SET OUTPUT WEIGHTS TO STANDARD DEVIATION OF sqrt of input_count
        wmo.resize(middle_count, vector<ld> (output_count));
        bmo.resize(output_count);
        for(int i = 0; i < middle_count; i++){
            for(int j = 0; j < output_count; j++){
                wmo[i][j] = output_distribution(generator);
            }
        }
    };


    /*
    SIGMOID ACTIVATION FUNCTION
    
    http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    (Could perhaps give overflow errors if +-x > 500 ?!)
    */
    ld sigmoid(ld x){
        if(x >= 0)return 1 / (1 + exp(-x));
        return exp(x) / (1 + exp(x));
    }

    vector<ld> sigmoid_vec(vector<ld> &input){
        int n = input.size();
        vector<ld> r (n);

        for(int j = 0; j < n; j++){
            r[j] = sigmoid(input[j]);
        }

        return r;
    }


    vector<vector<ld>> sigmoid_mat(vector<vector<ld>> &input){
        int n = input.size();
        int m = input[0].size();
        vector<vector<ld>> r (n, vector<ld> (m));

        for(int j = 0; j < n; j++){
            for(int k = 0; k < m; k++){
                r[j][k] = sigmoid(input[j][k]);
            }
        }

        return r;
    }

    /*
    DERIVATIVE OF SIGMOID
    */
    ld dsigmoid(ld x){
        return sigmoid(x) * (1 - sigmoid(x));
    }

    /*
    DERIVATIVE OF SIGMOID OF A VECTOR
    */
    vector<ld> dsigmoid_vec(vector<ld> &y){
        int n = y.size();
        vector<ld> r (n);

        for(int i = 0; i < n; i++){
            r[i] = dsigmoid(y[i]);
        }

        return r;
    }

    vector<vector<ld>> dsigmoid_mat(vector<vector<ld>> &input){
        int n = input.size();
        int m = input[0].size();
        vector<vector<ld>> r (n, vector<ld> (m));

        for(int j = 0; j < n; j++){
            for(int k = 0; k < m; k++){
                r[j][k] = dsigmoid(input[j][k]);
            }
        }

        return r;
    }

    /*
    Softmax (exp-normalize) a vector, instead of sigmoid for last layer
    */
    vector<ld> softmax(vector<ld> x){
        int n = x.size();
        vector<ld> r (n);
        ld high = x[0];

        // GET HIGHEST NUM IN x
        for(int i = 1; i < n; i++){
            if(x[i] > high)high = x[i];
        }

        // SUM THE EXPONENT DIFFERENCE WITH high
        ld expSum = 0;
        for(int i = 0; i < n; i++){
            r[i] = exp(x[i] - high);
            expSum += r[i];
        }

        for(int i = 0; i < n; i++){
            r[i] /= expSum;
        }

        return r;
    }

    /*
    LOADS A TRAINING IMAGE AS A 28*28 VECTOR, TAKEN FROM (index) IN TRAINING SET (0-60000)
    */
    void load_image(int index){
        //if(label.size() != 10)label.resize(10,0);
        for(int i = 0; i < 10; i++)label[i] = 0;

        if(input.size() != 28*28)input.resize(28*28);
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                input[i*28+j] = image[index][i][j] != 0;
            }
        }
        label[image_label[index]] = 1;
        //cout << "SET LABEL TO : " << image_label[index] << "\n";
    }

    /*
    LOADS A TEST IMAGE AS A 28*28 VECTOR, TAKEN FROM (index) IN TEST SET (0-10000)
    */
    void load_test(int index){
        //if(label.size()!=10)label.resize(10);
        for(int i = 0; i < 10; i++)label[i] = 0;

        if(input.size() != 28*28)input.resize(28*28);
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                input[i*28+j] = test[index][i][j] != 0;
            }
        }
        label[test_label[index]] = 1;
    }


    /*
    RETURNS INDEX OF A RANDOM TRAINING BATCH    
    */
    int rand_batch(){
        int pos = (int) ((ld)(num) * (rand() / (RAND_MAX + 1.0)));

        return pos;
    }

    vector<ld> out_to_one_hot(vector<ld> &input){
        int n = input.size();
        vector<ld> r (n, 0);

        ld high = input[0];
        int pos = 0;
        for(int i = 1; i < n; i++){
            if(input[i] > high){
                pos = i;
                high = input[i];
            }
        }
        r[pos] = 1;

        return r;
    }

    ld calc_loss(vector<ld> &output, vector<ld> &target){
        ld loss = 0;
        int n = output.size();

        for(int i = 0; i < n; i++){
            //cout << "calc_loss " << output[i] << " - " << target[i] << "\n";
            loss += output[i] - target[i];
        }

        return loss;
    }

    ld cross_entropy_loss(vector<ld> yHat, vector<ld> y){
        ld sum = 0;
        int n = y.size();

        for(int i = 0; i < n; i++){
            sum -= y[i] * log(yHat[i]);
        }

        return sum;
    }

    vector<ld> find_cross_entropy(vector<ld> &output, vector<ld> &target){
        int n = output.size();
        vector<ld> r (n);

        for(int i = 0; i < n; i++){
            r[i] = -target[i] * log(output[i]);
        }

        return r;
    }

    vector<ld> find_errors(vector<ld> &output, vector<ld> &target){
        int n = output.size();
        vector<ld> r (n);

        for(int i = 0; i < n; i++){
            r[i] = output[i] - target[i];
        }

        return r;
    }

    bool is_correct(vector<ld> &one_hot_guess, vector<ld> &target){
        int n = one_hot_guess.size();

        for(int i = 0; i < n; i++){
            if(one_hot_guess[i] == 1 && target[i] == 1)return true;
            if(target[i] == 1)return false;
            if(one_hot_guess[i] == 1)return false;
        }

        return false;
    }

    void save_guess(vector<ld> &output){
        int n = output.size();
        vector<ld> one_hot_guess = out_to_one_hot(output);

        for(int i = 0; i < n; i++){
            if(one_hot_guess[i] == 1)guess = i;
        }
    }

    vector<ld> forward(){
        vector<vector<ld>> hidden_inputs = mat_add(
            mat_mult(vec_to_col(input), wih),
            vec_to_col(bih)
        );

        vector<vector<ld>> hidden_outputs = sigmoid_mat(hidden_inputs);

        /*START MIDDLE*/
        vector<vector<ld>> middle_inputs = mat_add(
            mat_mult(hidden_outputs, whm),
            vec_to_col(bhm)
        );

        vector<vector<ld>> middle_outputs = sigmoid_mat(middle_inputs);
        /*END MIDDLE*/

        vector<vector<ld>> final_inputs = mat_add(
            mat_mult(middle_outputs, wmo, false),
            vec_to_col(bmo)
        );

        //vector<vector<ld>> final_outputs = sigmoid_mat(final_inputs);

        //vector<ld> yj = final_outputs[0];
        vector<ld> yj = softmax(final_inputs[0]);


        return yj;
    }


    /*
    INSPIRATION FROM https://www.pycodemates.com/2023/04/coding-a-neural-network-from-scratch-using-python.html
    */
    ld backprop(){
        bool comment = false;

        vector<vector<ld>> hidden_inputs = mat_add(
            mat_mult(vec_to_col(input), wih),
            vec_to_col(bih)
        );

        vector<vector<ld>> hidden_outputs = sigmoid_mat(hidden_inputs);

        /*START MIDDLE*/
        vector<vector<ld>> middle_inputs = mat_add(
            mat_mult(hidden_outputs, whm),
            vec_to_col(bhm)
        );

        vector<vector<ld>> middle_outputs = sigmoid_mat(middle_inputs);
        /*END MIDDLE*/

        vector<vector<ld>> final_inputs = mat_add(
            mat_mult(middle_outputs, wmo, false),
            vec_to_col(bmo)
        );

        //vector<vector<ld>> final_outputs = sigmoid_mat(final_inputs);
        vector<ld> final_outputs = softmax(final_inputs[0]);

        //vector<ld> yj = final_outputs[0];
        vector<ld> yj = final_outputs;

        save_guess(yj);

        /* STARTING BACKPROP */
        ld output_loss = cross_entropy_loss(yj, label);
        
        /*for(auto x : label)cout << x << " ";
        cout << "\n";*/
        //ld output_loss = calc_loss(yj, label);

        /* READ ERRORS PER LAYER */
        //vector<ld> output_errors = find_cross_entropy(yj, label);
        vector<ld> output_errors = find_errors(yj, label);

        vector<ld> middle_errors = mat_mult(
            vec_to_col(output_errors),
            transpose(wmo)
        )[0];

        vector<ld> hidden_errors = mat_mult(
            vec_to_col(middle_errors),
            transpose(whm)
        )[0];

        // CALCULATE W GRADIENTS
        vector<vector<ld>> wmo_g = mat_mult(
            transpose(middle_outputs),
            vec_to_col(pair_mult(
                output_errors, 
                yj
            ))
        );
        //dsigmoid_vec(yj)

        vector<vector<ld>> whm_g = mat_mult(
            transpose(hidden_outputs),
            vec_to_col(pair_mult(
                middle_errors, 
                dsigmoid_vec(middle_outputs[0])
            ))
        );
        
        vector<vector<ld>> wih_g = mat_mult(
            vec_to_row(input),
            vec_to_col(pair_mult(
                hidden_errors, 
                dsigmoid_vec(hidden_outputs[0])
            ))
        );
        
        // WRITE W GRAIDENT TO W
        for(int i = 0; i < middle_count; i++){
            for(int j = 0; j < output_count; j++){
                wmo[i][j] -= lr * wmo_g[i][j];
            }
        }
        for(int i = 0; i < hidden_count; i++){
            for(int j = 0; j < middle_count; j++){
                whm[i][j] -= lr * whm_g[i][j];
            }
        }
        for(int i = 0; i < input_count; i++){
            for(int j = 0; j < hidden_count; j++){
                wih[i][j] -= lr * wih_g[i][j];
            }
        }

        // CALCULATE B GRADIENTS
        vector<ld> bmo_g = pair_mult(
            output_errors,
            yj
        );
        //dsigmoid_vec(yj)
        
        vector<ld> bhm_g = pair_mult(
            middle_errors, 
            dsigmoid_vec(middle_outputs[0])
        );
        
        vector<ld> bih_g = pair_mult(
            hidden_errors, 
            dsigmoid_vec(hidden_outputs[0])
        );

        /* WRITE B GRADIENTS TO B */
        for(int i = 0; i < output_count; i++){
            bmo[i] -= lr * bmo_g[i];
        }
        for(int i = 0; i < middle_count; i++){
            bhm[i] -= lr * bhm_g[i];
        }
        for(int i = 0; i < hidden_count; i++){
            bih[i] -= lr * bih_g[i];
        }

        return output_loss;
    }

    ld mse(){
        int sum = 0;

        /*for(int i = 0; i < BATCH_SIZE; i++){
            if(one_hot_out[i][label[i]] != 1)sum++;
        }

        return (ld)sum / (ld)BATCH_SIZE;*/
        return 0;
    }

    /*
    TESTS THE PROGRAM ON HALF THE TESTING DATA
    */
    ld test_program(int half=0){
        int offset = 0;
        if(half)offset = 5000;

        int wrong = 0;
        for(int i = offset; i < 5000 + offset; i++){
            load_test(i);
            vector<ld> guess = forward();
            if(i == offset){
                for(auto x : guess)cout << x << " ";
                cout << "\n";
            }
            vector<ld> guess_one_hot = out_to_one_hot(guess);
            if(!is_correct(guess_one_hot, label))wrong++;
            if(i%100==0){
                for(auto x : guess_one_hot)cout << x << " ";
                cout << "\n";
                for(auto y : label)cout << y << " ";
                cout << "\n";
                cout << "TEST: " << i-offset << " -> ";
                cout << ((ld)wrong/(ld)(i-offset+1)) << "\n";
            }
        }

        cout << "TESTED: 5000/5000" << " -> ";
        cout << ((ld)wrong/(ld)(5000)) << "\n";

        return (ld)wrong/(ld)(5000);
    }

    ld forward_sample(){
        load_image(0);
        forward();
        return 0;
    }

    ld train_sample(){
        load_image(0);
        cout << "LOSS: " << backprop() << "\n";
        return 0;
    } 

    void train(){
        for(int e = 0; e < 50; e++){
            for(int i = (e%10)*6000; i < 6000 + (e%10)*6000; i++){
                load_image(i);
                ld loss = backprop();
                if(i%200==0)cout << "e" << e << "." << i << " -> " << guess << "-> LOSS: " << loss << "\n";
            }
            test_program();
        }
    }

};

int main(){
    input();

    ML ml = ML();

    ml.train();
    //ml.test_program();
    //ml.train_sample();
    //ml.forward_sample();

    return 0;
}