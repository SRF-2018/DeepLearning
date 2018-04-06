#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>
#include <iomanip>
using namespace std;
const string test_image = "/Users/gangseong-yong/Desktop/KangSungYong/train2/test_image.in";
const string test_label = "/Users/gangseong-yong/Desktop/KangSungYong/train2/test_label.in";
const string model_mat = "/Users/gangseong-yong/Desktop/KangSungYong/train2/mat.in";

const int MAXN = 500;
const int width = 8;
const int height = 8;
const int n1 = width * height; // input layer
const int n2 = 128; // hidden layer
const int n3 = 10; // output layer (class)
const double lr = 1e-3; //learning rate
const double momentum = 0.9; // 모멘텀
const double eps = 1e-4;
double w1[MAXN][MAXN],delta1[MAXN][MAXN],out1[MAXN];
double w2[MAXN][MAXN],delta2[MAXN][MAXN],in2[MAXN],out2[MAXN],theta2[MAXN];
double in3[MAXN],out3[MAXN],theta3[MAXN];
double expect[MAXN];
int d[height + 1][width+ 1];
int accept = 0;
ifstream image;
ifstream label;

void init(){
    ifstream mat;
    mat.open(model_mat.c_str());
    for(int i = 0; i < n1; i++) for(int j = 0; j < n2; j++) mat >> w1[i][j];
    for(int i = 0; i < n2; i++) for(int j = 0; j < n3; j++) mat >> w2[i][j];
    mat.close();
}
double sigmoid(double x){ return 1.0 / (1.0 + exp(-x)); }
void perceptron(){
    for(int i = 0; i < n2; i++) in2[i] = 0.0;
    for(int i = 0; i < n3; i++) in3[i] = 0.0;
    for(int i = 0; i < n1; i++) for(int j = 0; j < n2; j++) in2[j] += out1[i] * w1[i][j];
    for(int i = 0; i < n2; i++) out2[i] = sigmoid(in2[i]);
    for(int i = 0; i < n2; i++) for(int j = 0; j < n3; j++) in3[j] += out2[i] * w2[i][j];
    for(int i = 0; i < n3; i++) out3[i] = sigmoid(in3[i]);
}
void input(){
    int num;
    for(int j = 0; j < height; j++) for(int i = 0; i < width; i++){
        image >> d[i][j];
        out1[i + j * width] = d[i][j];
    }
    label >> num;
    for(int i = 0; i < n3; i++) expect[i] = 0.0;
    expect[num] = 1.0;
    perceptron();
    double mx = -1e9; int piv = -1;
    for(int i = 0; i < n3; i++) if(mx < out3[i]){
        mx = out3[i];
        piv = i;
    }
    accept += (num == piv);
    cout << "Label " << num <<" predict " << piv << " " << out3[piv]<<endl;
    if(out3[piv] > 0.999){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                cout<<d[i][j]<<' ';
            }
            cout<<endl;
        }
    }
}
int main(){
    image.open(test_image.c_str());
    label.open(test_label.c_str());
    init();
    int test = 8000;
    for(int sample = 1; sample <= test; ++ sample){
        input();
    }
    cout<<"accuracy: "<< 1.0 * accept / test << endl;
    image.close();
    label.close();
}

