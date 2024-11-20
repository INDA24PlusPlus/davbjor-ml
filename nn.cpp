#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define ld long double

// ERROR - ON BACKTRACKING TO FIRST STEP - 0 ==> MATRIX 32 vs 784

// https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
const int MAXN = 6e4 + 7;
unsigned int image[MAXN][30][30];
unsigned int num, magic, rows, cols;
unsigned int image_label[MAXN];
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
}

void multiply(vector<vector<ld>> &A, vector<vector<ld>> &B, vector<vector<ld>> &C, bool comment = false){
    int n = A.size();
    int k = B.size();
    int m = B[0].size();
    // A = n rows, k elements;
    // B = k rows, m elements;
    // C = n rows, m elements;

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
}

vector<vector<ld>> mat_mult(vector<vector<ld>> *A, vector<vector<ld>> *B, bool comment = false){
    int n = A->size();
    int k = B->size();
    int m = (*B)[0].size();
    // A = n rows, k elements;
    // B = k rows, m elements;
    // C = n rows, m elements;
    vector<vector<ld>> C (n, vector<ld> (m));

    if(comment){
        cout << "A: " << A->size() << " " << (*A)[0].size() << "\n";
        cout << "B: " << B->size() << " " << (*B)[0].size() << "\n";
        cout << "C: " << C.size() << " " << C[0].size() << "\n";
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            // index i,j in C, determined by:
                //sum of (i,k in A) * (k, j in B)
            C[i][j] = 0;
            for(int x = 0; x < k; x++){
                C[i][j] += (ld)(*A)[i][x] * (*B)[x][j];
            }
        }
    }

    return C;
}

vector<ld> pair_mult(vector<ld>* A, vector<ld>* B){
    int n = A->size();
    if(B->size() != n){
        cout << "WRONG PAIR MULT";
        return *A;
    }
    vector<ld> r (n);

    for(int i = 0; i < n; i++){
        r[i] = (*A)[i] * (*B)[i];
    }

    return r;
}

void add_bias(vector<vector<ld>> &A, vector<ld> &B){
    int n = A.size();
    int m = B.size();
    // A = n rows, m elements;
    // B = m elements

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            A[i][j] += B[j];
        }
    }
}

/*
SUB 2D VECTORS
*/
vector<vector<ld>> mat_sub(vector<vector<ld>>* A, vector<vector<ld>>* B){
    int n = A->size();
    int m = (*A)[0].size();
    if(n != B->size())return *A;
    if(m != (*B)[0].size())return *A;
    vector<vector<ld>> R (n, vector<ld> (m));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            R[i][j] = (*A)[i][j] - (*B)[i][j];
        }
    }

    return R;
}

/*
TRANSPOSE A
*/
vector<vector<ld>> transpose(vector<vector<ld>>* A){
    int n = A->size();
    int m = (*A)[0].size();
    vector<vector<ld>> R (m, vector<ld> (n));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            R[j][i] = (*A)[i][j];
        }
    }

    return R;
}

/*
MAKE VECTOR INTO 2D ARRAY WITH VALUES IN THE COLUMN
*/
vector<vector<ld>> vec_to_col(vector<ld>* A){
    int n = A->size();
    vector<vector<ld>> R (1, vector<ld> (n));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < 1; j++){
            R[j][i] = (*A)[i];
        }
    }

    return R;
}


/*
MAKE VECTOR INTO 2D ARRAY WITH VALUES IN THE ROW
*/
vector<vector<ld>> vec_to_row(vector<ld>* A){
    int n = A->size();
    vector<vector<ld>> R (n, vector<ld> (1));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < 1; j++){
            R[i][j] = (*A)[i];
        }
    }

    return R;
}

struct NN {
    int BATCH_SIZE;
    ld LEARNING_RATE = 0.0005;
    vector<int> LAYERS;

    // [BATCH_SIZE][NODE_COUNT]
    vector<vector<ld>> sample;

    // [BATCH_SIZE]
    vector<ld> label;

    // [LAYERS][INPUT_COUNT][OUTPUT_COUNT]
    vector<vector<vector<ld>>> w;

    // [LAYERS][OUTPUT_COUNT]
    vector<vector<ld>> b;

    // [LAYERS][BATCH_SIZE][INPUT_COUNT]
    vector<vector<vector<ld>>> activations {};

    // [BATCH_SIZE][OUTPUT_COUNT]
    vector<vector<ld>> one_hot_out;

    NN() {};
    NN(int b) : BATCH_SIZE(b) {};


    /*
    SIGMOID ACTIVATION FUNCTION
    
    http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    (Could perhaps give overflow errors if +-x > 500 ?!)
    */
    ld sigmoid(ld x){
        if(x >= 0)return 1 / (1 + exp(-x));
        return exp(x) / (1 + exp(x));
    }

    vector<vector<ld>> sigmoid_batch(vector<vector<ld>> input){
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
    vector<ld> dsigmoid_vec(vector<ld> y){
        int n = y.size();
        vector<ld> r (n);

        for(int i = 0; i < n; i++){
            r[i] = dsigmoid(y[i]);
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


    vector<vector<ld>> softmax_batch(vector<vector<ld>> input){
        int n = input.size();
        int m = input[0].size();

        vector<vector<ld>> r (n, vector<ld> (m));
        
        for(int i = 0; i < n; i++){
            r[i] = softmax(input[i]);
        }

        return r;
    }


    ld cross_entropy_loss(vector<ld> y, vector<ld> yHat){
        ld sum = 0;
        for(int i = 0; i < y.size(); i++){
            sum -= y[i] * log(yHat[i]);
        }

        return sum;
    }

    /*
    LOADS A BATCH of (BATCH_SIZE) FROM (index) IN TRAINING SET
    */
    void load_batch(int index){
        if(LAYERS.empty())LAYERS.push_back(28 * 28);
        if(sample.empty())sample = vector<vector<ld>> (BATCH_SIZE, vector<ld> (28 * 28));
        if(label.empty())label = vector<ld> (BATCH_SIZE, 0);

        for(int pos = 0; pos < BATCH_SIZE; pos++){
            for(int i = 0; i < 28; i++){
                for(int j = 0; j < 28; j++){
                    sample[pos][i*28+j] = image[pos + index][i][j] != 0;
                }
            }
            label[pos] = image_label[pos + index];
        }
    }
    void load_label_batch(int index){
        if(LAYERS.empty())LAYERS.push_back(10);
        if(sample.empty())sample = vector<vector<ld>> (BATCH_SIZE, vector<ld> (10));
        if(label.empty())label = vector<ld> (BATCH_SIZE, 0);

        for(int pos = 0; pos < BATCH_SIZE; pos++){
            label[pos] = image_label[pos + index];
        }
        vector<vector<ld>> label_one_hot = label_to_one_hot(label, 10);
        for(int i = 0; i < BATCH_SIZE; i++){
            sample[i] = label_one_hot[i];
        }
    }

    /*
    LOAD LABEL BATCH
    */

    /*
    RETURNS INDEX OF A RANDOM TRAINING BATCH    
    */
    int rand_batch(){
        int pos = (int) ((ld)(num - BATCH_SIZE) * (rand() / (RAND_MAX + 1.0)));

        return pos;
    }

    /*
    ADDS A LAYER TO NN
    
    CREATES RANDOM weights for every edge
        uses standard deviation of sqrt of input nodes

    CREATES ZEROED biases for every node
    */
    void add_layer(int output_count){
        // CREATE FIRST INPUT LAYER IF IT DOESN'T EXIST
        if(LAYERS.empty())LAYERS.push_back(28 * 28);

        // PUSH NEW LAYER
        int input_count = LAYERS[LAYERS.size()-1];
        LAYERS.push_back(output_count);

        default_random_engine generator;
        normal_distribution<long double> distribution(0, 1 / sqrt(input_count));

        int l = w.size();
        w.push_back(vector<vector<ld>> (input_count, vector<ld> (output_count)));
        b.push_back(vector<ld> (vector<ld> (output_count, 0)));

        // SET WEIGHTS TO STANDARD DEVIATION OF sqrt of input_count
        for(int i = 0; i < input_count; i++){
            for(int j = 0; j < output_count; j++){
                w[l][i][j] = distribution(generator);
            }
        }
    }

    // CONVERT BATCH OF LABELS TO BATCH OF ONE HOT ARRAYS
    vector<vector<ld>> label_to_one_hot(vector<ld> y, int n){
        vector<vector<ld>> r (y.size(), vector<ld> (n, 0));

        for(int i = 0; i < y.size(); i++){
            r[i][y[i]] = 1;
        }

        return r;
    }

    vector<vector<ld>> out_to_one_hot(vector<vector<ld>> input){
        int n = input.size();
        int m = input[0].size();
        vector<vector<ld>> r (n, vector<ld> (m));

        
        for(int i = 0; i < n; i++){
            ld high = input[i][0];
            int pos = 0;
            for(int j = 1; j < m; j++){
                if(input[i][j] > high){
                    pos = j;
                    high = input[i][j];
                }
            }
            
            r[i][pos] = 1.0;
        }

        return r;
    }

    ld calc_loss(vector<vector<ld>> x, vector<vector<ld>> y){
        ld loss = 0;

        for(int i = 0; i < x.size(); i++){
            loss += cross_entropy_loss(y[i], x[i]);
        }
        loss /= (ld)BATCH_SIZE;

        return loss;
    }

    ld mse(){
        int sum = 0;

        for(int i = 0; i < BATCH_SIZE; i++){
            if(one_hot_out[i][label[i]] != 1)sum++;
        }

        return (ld)sum / (ld)BATCH_SIZE;
    }

    ld forward_sample(){
        // [BATCH_SIZE, INPUT_COUNT]
        vector<vector<ld>> a = sample;

        for(int i = 0; i < w.size(); i++){
            // [BATCH_SIZE][OUTPUT_COUNT]
            vector<vector<ld>> z (BATCH_SIZE, vector<ld> (LAYERS[i+1]));

            multiply(a, w[i], z);
            add_bias(z, b[i]);

            if(i == w.size()-1)
                a = softmax_batch(z);
            else
                a = sigmoid_batch(z);
        }

        vector<vector<ld>> one_hot_y = label_to_one_hot(label, 10);
        one_hot_out = out_to_one_hot(a);
    
        ld loss = calc_loss(a, one_hot_y);

        return loss;
    }

    void forward_dataset(){
        for(int i = 0; i < 10; i++){
            load_batch(rand_batch());
            ld loss = forward_sample();
            cout << "LOSS: " << loss << "\n";
            cout << "MSE : " << mse() << "\n";
        }

    }

    ld train_sample(){
        // [BATCH_SIZE, INPUT_COUNT]
        vector<vector<ld>> a = sample;

        for(int i = 0; i < w.size(); i++){
            // [BATCH_SIZE][OUTPUT_COUNT]
            vector<vector<ld>> z (BATCH_SIZE, vector<ld> (LAYERS[i+1]));

            multiply(a, w[i], z);
            add_bias(z, b[i]);

            if(i == w.size()-1)
                a = softmax_batch(z);
            else
                a = sigmoid_batch(z);
        
            activations.push_back(a);
        }

        // [BATCH_SIZE][10]
        vector<vector<ld>> one_hot_y = label_to_one_hot(label, 10);
        // [BATCH_SIZE][10]
        one_hot_out = out_to_one_hot(a);
    
        ld loss = calc_loss(a, one_hot_y);


        /*
        START BACKPROPAGATION
        */
             
        // W_G = [layer][input_count][node_count] - weight gradient
        vector<vector<vector<ld>>> w_g (w.size(), vector<vector<ld>> {});

        // B_G = [layer][node_count] - bias gradient
        vector<vector<ld>> b_g (b.size(), vector<ld> (b[0].size()));

        // A_G = [layer][node_count] - activation gradient
        vector<vector<ld>> a_g (b.size(), vector<ld> (b[0].size()));

        // Go through layers in reverse
        for(int i = w.size() - 1; i >= 0; i--){
            bool is_last = i == w.size() - 1;
            bool is_second_last = i == w.size() - 2;

            for(int j = 0; j < BATCH_SIZE; j++){
                if(is_last){
                    vector<vector<ld>> y = vec_to_col(&one_hot_y[j]);
                    vector<vector<ld>> a = vec_to_col(&activations[i][j]);
                    vector<vector<ld>> a_prev = vec_to_col(&activations[i-1][j]);

                    vector<vector<ld>> left = transpose(&a_prev);
                    vector<vector<ld>> right = mat_sub(&a, &y);


                    w_g[i] = mat_mult(&left, &right, false);
                    b_g[i] = transpose(&right)[0];
                }
                else {
                    vector<vector<ld>> y = vec_to_col(&one_hot_y[j]);
                    vector<vector<ld>> a = vec_to_col(&activations[i][j]);
                    vector<vector<ld>> w_next = w[i+1];
                    vector<vector<ld>> a_next = vec_to_col(&activations[i+1][j]);

                    if(i > 0)
                        vector<vector<ld>> a_prev = vec_to_col(&activations[i-1][j]);
                    else
                        vector<vector<ld>> a_prev = vec_to_col(&sample[j]);

                    if(is_second_last){
                        vector<vector<ld>> left = mat_sub(&a_next, &y);
                        vector<vector<ld>> right = transpose(&w_next);
                        
                        vector<ld> dCda = mat_mult(&left, &right, false)[0];
                        a_g[i] = dCda;
                    }
                    else {
                        vector<ld> dCda_next = a_g[i+1];

                        // dsigmoid(a_next) * dCda_next
                        vector<ld> dsig = dsigmoid_vec(a_next[0]);
                        vector<ld> pm = pair_mult(&dsig, &dCda_next);
                        vector<vector<ld>> left = vec_to_col(&pm);
                        vector<vector<ld>> right = transpose(&w_next);

                        vector<ld> dCda = mat_mult(&left, &right, false)[0];
                        a_g[i] = dCda;
                    }

                    vector<ld> temp_a = dsigmoid_vec(a[0]);
                    vector<ld> x = pair_mult(&temp_a, &a_g[i]);
                    vector<vector<ld>> a_mat = vec_to_col(&temp_a);

                    vector<vector<ld>> left = transpose(&a_mat);
                    vector<vector<ld>> right = vec_to_col(&x);
    
                    w_g[i] = mat_mult(&left, &right, true);
                    b_g[i] = x;
                    
                }

                /* UPDATE W */
                cout << i << " ==> " << w_g[i].size() << " vs" << w[i].size() << " && " << w_g[i][0].size() << " vs " << w[i][0].size() << "\n";
                for(int k = 0; k < w_g[i].size(); k++){
                    for(int l = 0; l < w_g[i][0].size(); l++){
                        bool is_nan = isnan(w_g[i][k][l]);
                        //if(is_nan)cout << "is_nan\n";
                        //if(!is_nan)cout << "TUNE W " << i << " " << w_g[i][k][l] << "\n";
                        if(!is_nan)w[i][k][l] -= w_g[i][k][l] * LEARNING_RATE;
                    }
                }
                /* UPDATE B */
                for(int k = 0; k < b_g[i].size(); k++){
                    bool is_nan = isnan(b_g[i][k]);
                    //if(is_nan)cout << "is_nan\n";
                    //if(!is_nan)cout << "TUNE B " << i << " " << b_g[i][k] << "\n";
                    if(!is_nan)b[i][k] -= b_g[i][k] * LEARNING_RATE;
                }
            }


        }
       return loss;
    }

    void train(){
        ld best_mse = 1;
        ld best_loss = 10;
        for(int i = 0; i < 1000; i++){
            //
            load_batch(rand_batch());
            //load_label_batch(rand_batch());
            ld loss = train_sample();
            ld mean_error = mse();
            cout << i << " LOSS: " << loss << "\n";
            cout << i << " MSE : " << mean_error << "\n";

            best_loss = min(best_loss, loss);
            best_mse = min(best_mse, mean_error);

            for(int j = 0; j < min(6,BATCH_SIZE); j++){
                cout << label[j] << " -> ";
                for(auto x : one_hot_out[j])cout << x << " ";
                cout << "\n";
            }
        }
        cout << "BEST MSE : " << best_mse << "\n";
        cout << "BEST LOSS : " << best_loss << "\n";
    }

};

int main(){
    input();

    NN nn = NN(64);
    nn.add_layer(32);
    nn.add_layer(32);
    nn.add_layer(10);

    nn.train();

    return 0;
}