#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define ld long double

// https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
const int MAXN = 6e4 + 7;
unsigned int image[MAXN][30][30];
unsigned int num, magic, rows, cols;
unsigned int label[MAXN];
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
        label[i] = in(icin, 1);
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

struct NN {
    int BATCH_SIZE;
    ld LEARNING_RATE = 0.01;

    // X = [layer][batch_count][node_count] - input data at every node
    vector<vector<vector<ld>>> x {};

    // W = [layer][input_count][node_count] - weight of every edge
    vector<vector<vector<ld>>> w {};

    // B = [layer][node_count] - weight of every node's bias
    vector<vector<ld>> b {};

    // Y = [batch_count] - labels for entire batch
    vector<int> y {};

    // OUT = [batch_count] - output for entire batch
    vector<int> out {};



    // BEST_W = [layer][input_count][node_count] - weight of every edge
    vector<vector<vector<ld>>> best_w {};

    // BEST_B = [layer][node_count] - weight of every node's bias
    vector<vector<ld>> best_b {};

    // BEST ERR
    ld best_err;



    NN() {};
    NN(int b) : BATCH_SIZE(b) {};

    /*
    LOADS A BATCH of (BATCH_SIZE) FROM (index) IN TRAINING SET
    */
    void load_batch(int index){
        // X0 = 
        if(x.empty())x = vector<vector<vector<ld>>> {vector<vector<ld>> (BATCH_SIZE, vector<ld> (28 * 28))};
        y = vector<int> (BATCH_SIZE, 0);

        for(int pos = 0; pos < BATCH_SIZE; pos++){
            for(int i = 0; i < 28; i++){
                for(int j = 0; j < 28; j++){
                    x[0][pos][i*28+j] = image[pos + index][i][j] != 0;
                }
            }
            y[pos] = label[pos + index];
        }

        /*cout << "LABELS\n";
        for(int i = 0; i < BATCH_SIZE; i++){
            cout << y[i] << "\n";
        }*/
    }

    /*
    ADDS A LAYER TO NN

    SETS INPUT COUNT TO LAST X's INNER SIZE (or 28*28 for first)
    
    CREATES RANDOM weights for every edge
        uses standard deviation of sqrt of input nodes

    CREATES ZEROED biases for every node


    */
    void add_layer(int node_count){
        // CREATE FIRST INPUT LAYER IF IT DOESN'T EXIST
        if(x.empty())x.push_back(vector<vector<ld>> (BATCH_SIZE, vector<ld> (28 * 28, 0)));
        

        int input_count = x[x.size()-1][0].size();

        default_random_engine generator;
        normal_distribution<long double> distribution(0, 1 / sqrt(input_count));

        int index = w.size();
        w.push_back(vector<vector<ld>> (input_count, vector<ld> (node_count)));
        b.push_back(vector<ld> (vector<ld> (node_count)));

        for(int i = 0; i < input_count; i++){
            for(int j = 0; j < node_count; j++){
                w[index][i][j] = distribution(generator);
                //w[index][i][j] = (((ld)1)/(ld)(0.01 + rand() % 100));
                //w[index][i][j] = (((ld)1)/(ld)(0.01 + rand() % 100)) - 0.5;
            }
        }

        for(int j = 0; j < node_count; j++){
            //b[index][j] = (((ld)1)/(ld)(0.01 + rand() % 100));
            b[index][j] = 0;
        }

        //cout << "PUSHING TO X: " << x.size() << " -> " << BATCH_SIZE << " " << node_count << "\n";
        x.push_back(vector<vector<ld>> (BATCH_SIZE, vector<ld> (node_count, 0)));
    }

    /*
    SUM OF ALL ERRORS MADE IN BATCH
    
    */
    int sum_err(){
        int sum = 0;

        vector<int> res (10, 0);

        /*for(int i = 0; i < 10; i++){
            cout << x[x.size() - 1][0][i] << " ";
            cout << "\n";
        }*/


        for(int i = 0; i < BATCH_SIZE; i++){
            //cout << out[i] << " vs " << y[i] << "\n";
            res[out[i]] += 1;
            if(out[i] != y[i])sum += 1;
            //sum += abs((out[i] - y[i])) * abs((out[i] - y[i]));
        }

        for(auto x : res)cout << x << " ";
        cout << "\n";

        return sum;
    }

    /*
    NORMALIZE OUT (from vector of size 10 -> store the highest index of batch in out[batch_index] = high_index)
    
    */
    void one_hot_out(){
        int l = x.size() - 1;
        
        for(int i = 0; i < BATCH_SIZE; i++){
            ld high = x[l][i][0];
            int pos = 0;
            for(int j = 1; j < 10; j++){
                //cout << x[l][i][j] << " ";
                if(x[l][i][j] > high){
                    pos = j;
                    high = x[l][i][j];
                }
            }
            //cout << "FOUND: " << pos << "\n";
            if(out.size() == i)out.push_back(pos);
            else out[i] = pos;
        }
    }

    vector<ld> integer_to_one_hot(int t, int n){
        vector<ld> r (n, 0);

        r[t] = 1;
        return r;
    }

    ld predict(){
        //print_batch(1);
        for(int i = 0; i < w.size(); i++){
            // For every layer

            // X0 * W0 + B0 = Y1

            multiply(x[i], w[i], x[i+1]);
            add_bias(x[i+1], b[i]);
            
            bool is_last = i == w.size() - 1;
            if(is_last){
                // X_last = softmax(Y_last)
                for(int j = 0; j < x[i+1].size(); j++){
                    softmax(x[i+1][j]);
                }
                cout << "Guess:\n";
                for(int j = 0; j < 10; j++){
                    cout << x[i+1][0][j] << " ";
                }
                cout << "\n";
                break;
            }

            // X1 = f(Y1)
            for(int j = 0; j < x[i+1].size(); j++){
                for(int k = 0; k < x[i+1][0].size(); k++){
                    x[i+1][j][k] = sigmoid(x[i+1][j][k]);
                }
            }
        }

        vector<vector<ld>> one_hot_y (BATCH_SIZE, vector<ld> (10));
        for(int i = 0; i < BATCH_SIZE; i++)one_hot_y[i] = integer_to_one_hot(y[i], 10);
        
        ld loss = 0;
        for(int i = 0; i < BATCH_SIZE; i++)loss += cross_entropy_loss(one_hot_y[i], x[x.size()-1][i]);
        loss /= BATCH_SIZE;

        cout << "LOSS: " << loss << "\n";

        one_hot_out();



        /*
        BACKPROP
        https://github.com/marcospgp/backpropagation-from-scratch
        
        */

        // W_G = [layer][input_count][node_count] - weight of every edge
        vector<vector<vector<ld>>> w_g (w.size(), vector<vector<ld>> {});

        // B_G = [layer][node_count] - weight of every node's bias
        vector<vector<ld>> b_g (b.size(), vector<ld> (b[0].size()));

        for(int i = x.size() - 1; i >= 1; i--){
            bool is_last = i == w.size() - 1;
            if(is_last){
                // aT = A[batch_count][node_count]^T
                vector<vector<ld>> aT (10, vector<ld> (BATCH_SIZE));


                //cout << "fill b\n";
                // b[i-1] = x[i] - one_hot_y
                for(int k = 0; k < 10; k++){
                    for(int j = 0; j < BATCH_SIZE; j++){
                        aT[k][j] = x[i][j][k] - one_hot_y[j][k];

                        b_g[i][k] += aT[k][j];
                    }
                    b_g[i][k] /= BATCH_SIZE;
                }

                // w[i-1] = np.matmul((x[i] - one_hot_y), x[i-1].T)
                vector<vector<ld>> wT (w[i-1][0].size(), vector<ld> (w[i-1].size()));
                multiply(aT, x[i-1], wT);
                
                for(int j = 0; j < wT[0].size(); j++){
                    w_g[i-1].push_back(vector<ld> (wT.size()));

                    for(int k = 0; k < wT.size(); k++){
                        w_g[i-1][j][k] = wT[k][j];
                    }
                }


                /*
                UPDATE B AND W FROM GRADIENT
                */                

                cout << "Update B gradient:\n";
                for(int j = 0; j < 10; j++){
                    cout << b_g[i][j] << " ";
                    b[i][j] -= b_g[i][j] * LEARNING_RATE;
                }
                cout << "\n";

                for(int j = 0; j < w[i-1].size(); j++){
                    for(int k = 0; k < w[i-1][0].size(); k++){
                        w[i-1][j][k] -= w_g[i-1][j][k] * LEARNING_RATE;
                    }
                }
                continue;
            }




        }


        return loss;
    }


    /*
    SIGMOID ACTIVATION FUNCTION
    
    http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    (Could perhaps give overflow errors if +-x > 500 ?!)
    */
    ld sigmoid(ld x){
        if(x >= 0)return 1 / (1 + exp(x));
        return exp(x) / (1 + exp(x));
    }

    /*
    DERIVATIVE OF SIGMOID
    */
    ld dsigmoid(ld x){
        return sigmoid(x) * (1 - sigmoid(x));
    }

    /*
    Softmax (exp-normalize) a vector, instead of sigmoid for last layer
    
    */
    void softmax(vector<ld> &x){
        ld high = x[0];

        for(ld z : x){
            if(z > high)high = z;
        }

        ld expSum = 0;
        for(ld &z : x){
            z = exp(z - high);
            expSum += z;
        }

        for(ld &z : x){
            z /= expSum;
        }
    }


    ld cross_entropy_loss(vector<ld> y, vector<ld> yHat){
        ld sum = 0;
        for(int i = 0; i < y.size(); i++){
            sum -= y[i] * log(yHat[i]);
        }

        return sum;
    }

    ld mse(){

        ld loss = predict();
        int sum = sum_err();

        return (ld)sum / (ld)BATCH_SIZE;
    }


    void train(){
        cout << "NN size -> \n";
        for(auto nodes : x){
            cout << nodes[0].size() << "\n";
        }
        cout << "X size -> " << x.size() << "\n";
        cout << "W size -> " << w.size() << "\n";
        cout << "B size -> " << b.size() << "\n";

        ld best_err = 1;
        for(int i = 0; i < 30; i++){
            load_batch(rand_batch());
            best_err = min(best_err, mse());
        }
        cout << mse() << "\n";
    }


    void print_batch(int max_size = 10000){
        if(x.empty())return;

        for(int k = 0; k < min((int)x[0].size(), max_size); k++){
            for(int i = 0; i < 28; i++){
                for(int j = 0; j < 28; j++){
                    cout << x[0][k][i*28 + j];
                }
                cout << "\n";
            }
            cout << y[k] << "\n";
        }
    }

    void print_labels(){
        for(int k = 0; k < y.size(); k++){
            cout << y[k] << "\n";
        }
    }

    int rand_batch(){
        int pos = (int) ((ld)(num - BATCH_SIZE) * (rand() / (RAND_MAX + 1.0)));
        //cout << "pos: " << pos << "\n";
        return pos;
    }
};


int main(){
    input();
    
    
    int print_nums = 1;
    for(int k = 0; k < print_nums; k++){
        for(int i = 0; i < 30; i++){
            for(int j = 0; j < 30; j++){
                cout << (image[k][i][j] != 0);
            }
            cout << "\n";
        }
        cout << label[k] << "\n";
    }
   

    /*nn.load_batch(batch_size, rand_batch(batch_size));*/

    NN nn = NN(128);
    nn.add_layer(32);
    nn.add_layer(32);
    nn.add_layer(10);

    nn.train();



    //nn.print_batch();
    //nn.print_labels();
    

    return 0;
}