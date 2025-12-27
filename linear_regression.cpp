#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <vector>
using namespace std;

/* -------------------------
   Linked List for 2D Matrix
   ------------------------- */
struct Node {
    double* data;
    Node* next;
    Node(int cols) { data = new double[cols]; for(int i=0;i<cols;i++) data[i]=0; next=nullptr; }
};

struct MatrixList {
    Node* head;
    int rows;
    int cols;
    MatrixList(int r, int c) { rows=r; cols=c; head=nullptr; }
    void addRow(Node* n) {
        if(!head) { head=n; } 
        else { Node* t=head; while(t->next) t=t->next; t->next=n; }
    }
};

/* -------------------------
   Utility functions
   ------------------------- */
static inline string trim(const string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

void quickSort(double* arr, int low, int high) {
    if(low>=high) return;
    double pivot = arr[(low+high)/2];
    int i=low, j=high;
    while(i<=j){
        while(arr[i]<pivot) i++;
        while(arr[j]>pivot) j--;
        if(i<=j){ swap(arr[i],arr[j]); i++; j--; }
    }
    if(low<j) quickSort(arr,low,j);
    if(i<high) quickSort(arr,i,high);
}

/* -------------------------
   Matrix helpers
   ------------------------- */
double** create2DArray(int r, int c){
    double** arr = new double*[r];
    for(int i=0;i<r;i++) arr[i] = new double[c]{0};
    return arr;
}

double** transpose(double** A, int r, int c){
    double** At = create2DArray(c,r);
    for(int i=0;i<r;i++) for(int j=0;j<c;j++) At[j][i]=A[i][j];
    return At;
}

double** matMul(double** A, int rA,int cA, double** B, int rB,int cB){
    double** C = create2DArray(rA,cB);
    for(int i=0;i<rA;i++) for(int k=0;k<cA;k++) for(int j=0;j<cB;j++)
        C[i][j]+=A[i][k]*B[k][j];
    return C;
}

bool inverseMatrix(double** A, int n, double** inv){
    double** aug = create2DArray(n,2*n);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++) aug[i][j]=A[i][j];
        aug[i][n+i]=1.0;
    }
    for(int col=0;col<n;col++){
        int pivot=col;
        for(int r=col;r<n;r++) if(fabs(aug[r][col])>fabs(aug[pivot][col])) pivot=r;
        if(fabs(aug[pivot][col])<1e-12) aug[col][col]+=1e-8;
        if(pivot!=col) { for(int j=0;j<2*n;j++) swap(aug[pivot][j],aug[col][j]); }
        double val=aug[col][col];
        for(int j=0;j<2*n;j++) aug[col][j]/=val;
        for(int r=0;r<n;r++){
            if(r==col) continue;
            double factor=aug[r][col];
            for(int j=0;j<2*n;j++) aug[r][j]-=factor*aug[col][j];
        }
    }
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) inv[i][j]=aug[i][n+j];
    for(int i=0;i<n;i++) delete[] aug[i]; delete[] aug;
    return true;
}

/* -------------------------
   Regression
   ------------------------- */
bool multipleLinearRegression(double** X, double* y, int n, int p, double* beta){
    double** Xt = transpose(X,n,p);
    double** XtX = matMul(Xt,p,n,X,n,p);
    double** XtX_inv = create2DArray(p,p);
    if(!inverseMatrix(XtX,p,XtX_inv)) return false;

    double** yMat = create2DArray(n,1);
    for(int i=0;i<n;i++) yMat[i][0]=y[i];
    double** Xt_y = matMul(Xt,p,n,yMat,n,1);

    double** betaMat = matMul(XtX_inv,p,p,Xt_y,p,1);
    for(int i=0;i<p;i++) beta[i]=betaMat[i][0];

    for(int i=0;i<p;i++){ delete[] Xt[i]; delete[] XtX[i]; delete[] XtX_inv[i]; delete[] betaMat[i]; }
    for(int i=0;i<n;i++) delete[] yMat[i];
    delete[] Xt; delete[] XtX; delete[] XtX_inv; delete[] yMat; delete[] Xt_y; delete[] betaMat;
    return true;
}

double computeR2(double* y_true, double* y_pred, int n){
    double mean=0, ss_tot=0, ss_res=0;
    for(int i=0;i<n;i++) mean+=y_true[i];
    mean/=n;
    for(int i=0;i<n;i++){
        double diff=y_true[i]-y_pred[i];
        ss_res+=diff*diff;
        double diff2=y_true[i]-mean;
        ss_tot+=diff2*diff2;
    }
    if(fabs(ss_tot)<1e-12) return 0;
    return 1.0-(ss_res/ss_tot);
}

/* -------------------------
   Save / Load
   ------------------------- */
bool saveModel(const char* fname,double* beta,int p){
    ofstream of(fname,ios::trunc);
    if(!of.is_open()) return false;
    of<<p<<"\n";
    for(int i=0;i<p;i++){ if(i) of<<" "; of<<beta[i]; }
    of<<"\n"; of.close(); return true;
}

bool loadModel(const char* fname,double* beta,int &p){
    ifstream in(fname);
    if(!in.is_open()) return false;
    if(!(in>>p)) return false;
    for(int i=0;i<p;i++) if(!(in>>beta[i])) return false;
    return true;
}

/* -------------------------
   JSON Output Helpers
   ------------------------- */
void printTrainJSON(double* beta, int p, double r2){
    cout << "{\"betas\":[";
    for(int i=0;i<p;i++){
        cout << beta[i];
        if(i < p-1) cout << ",";
    }
    cout << "],\"r2\":" << r2 << ",\"status\":\"success\"}";
}

void printPredictJSON(double pred){
    cout << "{\"prediction\":" << pred << ",\"status\":\"success\"}";
}

/* -------------------------
   Main Program
   ------------------------- */
int main(int argc, char* argv[]){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    const char* modelFile="last_model.txt";

    // JSON Predict mode
    if(argc>=2 && string(argv[1])=="--jsonPredict"){
        double beta[100];
        int p=0;
        if(!loadModel(modelFile,beta,p)) { cerr<<"Error: Train first.\n"; return 1; }
        int expected=p-1;
        if(argc-2<expected){ cerr<<"Error: Provide "<<expected<<" inputs\n"; return 1; }
        double pred=beta[0];
        for(int i=0;i<expected;i++) pred+=beta[i+1]*stod(argv[i+2]);
        printPredictJSON(pred);
        return 0;
    }

    // JSON Train mode
    if(argc>=3 && string(argv[1])=="--jsonTrain"){
        ifstream fin(argv[2]);
        if(!fin.is_open()){ cerr<<"Error: Cannot open CSV\n"; return 1; }

        string line, cell;
        if(!getline(fin,line)){ cerr<<"CSV empty\n"; return 1; }

        int m=0; stringstream ss(line);
        while(getline(ss,cell,',')) m++;
        if(m<2){ cerr<<"Need >=1 input & 1 output\n"; return 1; }

        int n=0;
        fin.clear(); fin.seekg(0);
        getline(fin,line); // skip header
        while(getline(fin,line)) if(!line.empty()) n++;

        double** data = create2DArray(n,m);
        fin.clear(); fin.seekg(0); getline(fin,line);
        int r=0;
        while(getline(fin,line) && r<n){
            if(!line.empty() && line.back()=='\r') line.pop_back();
            stringstream ss2(line); int c=0;
            while(getline(ss2,cell,',') && c<m){
                string t=trim(cell);
                if(t.empty()) data[r][c]=0.0;
                else data[r][c]=stod(t);
                c++;
            }
            r++;
        }
        fin.close();

        for(int col=0;col<m;col++){
            double* tmp = new double[n];
            for(int i=0;i<n;i++) tmp[i]=data[i][col];
            quickSort(tmp,0,n-1);
            double median = (n%2==1)?tmp[n/2]:(tmp[n/2-1]+tmp[n/2])/2.0;
            for(int i=0;i<n;i++) if(data[i][col]==0) data[i][col]=median;
            delete[] tmp;
        }

        int inputCols=m-1;
        double** X = create2DArray(n,inputCols+1);
        double* y = new double[n];
        for(int i=0;i<n;i++){
            X[i][0]=1.0;
            for(int j=0;j<inputCols;j++) X[i][j+1]=data[i][j];
            y[i]=data[i][inputCols];
        }

        double* beta = new double[inputCols+1];
        if(!multipleLinearRegression(X,y,n,inputCols+1,beta)){ cerr<<"Regression failed\n"; return 1; }

        double* y_pred = new double[n];
        for(int i=0;i<n;i++){
            double p=beta[0];
            for(int j=0;j<inputCols;j++) p+=beta[j+1]*X[i][j+1];
            y_pred[i]=p;
        }

        double r2 = computeR2(y,y_pred,n);

        // JSON Output
        printTrainJSON(beta,inputCols+1,r2);

        if(!saveModel(modelFile,beta,inputCols+1)) cerr<<"Warning: Could not save model\n";

        for(int i=0;i<n;i++) delete[] data[i]; delete[] data;
        for(int i=0;i<n;i++) delete[] X[i]; delete[] X;
        delete[] y; delete[] beta; delete[] y_pred;
        return 0;
    }

    // Original modes (for backward compatibility)
    if(argc>=2 && string(argv[1])=="--predict"){
        double beta[100];
        int p=0;
        if(!loadModel(modelFile,beta,p)){ cerr<<"Error: Train first.\n"; return 1; }
        int expected=p-1;
        if(argc-2<expected){ cerr<<"Error: Provide "<<expected<<" inputs\n"; return 1; }
        double pred=beta[0];
        for(int i=0;i<expected;i++) pred+=beta[i+1]*stod(argv[i+2]);
        cout.setf(ios::fixed); cout.precision(6);
        cout<<pred<<"\n";
        return 0;
    }

    if(argc<2){ cerr<<"Usage: program <csv_path> OR program --predict val1 val2 ...\n"; return 1; }

    ifstream fin(argv[1]);
    if(!fin.is_open()){ cerr<<"Error: Cannot open CSV\n"; return 1; }

    string line, cell;
    if(!getline(fin,line)){ cerr<<"CSV empty\n"; return 1; }

    int m=0; stringstream ss(line);
    while(getline(ss,cell,',')) m++;
    if(m<2){ cerr<<"Need >=1 input & 1 output\n"; return 1; }

    int n=0;
    fin.clear(); fin.seekg(0);
    getline(fin,line);
    while(getline(fin,line)) if(!line.empty()) n++;

    double** data = create2DArray(n,m);
    fin.clear(); fin.seekg(0); getline(fin,line);
    int r=0;
    while(getline(fin,line) && r<n){
        if(!line.empty() && line.back()=='\r') line.pop_back();
        stringstream ss2(line); int c=0;
        while(getline(ss2,cell,',') && c<m){
            string t=trim(cell);
            if(t.empty()) data[r][c]=0.0;
            else data[r][c]=stod(t);
            c++;
        }
        r++;
    }
    fin.close();

    for(int col=0;col<m;col++){
        double* tmp = new double[n];
        for(int i=0;i<n;i++) tmp[i]=data[i][col];
        quickSort(tmp,0,n-1);
        double median = (n%2==1)?tmp[n/2]:(tmp[n/2-1]+tmp[n/2])/2.0;
        for(int i=0;i<n;i++) if(data[i][col]==0) data[i][col]=median;
        delete[] tmp;
    }

    int inputCols=m-1;
    double** X = create2DArray(n,inputCols+1);
    double* y = new double[n];
    for(int i=0;i<n;i++){
        X[i][0]=1.0;
        for(int j=0;j<inputCols;j++) X[i][j+1]=data[i][j];
        y[i]=data[i][inputCols];
    }

    double* beta = new double[inputCols+1];
    if(!multipleLinearRegression(X,y,n,inputCols+1,beta)){ cerr<<"Regression failed\n"; return 1; }

    double* y_pred = new double[n];
    for(int i=0;i<n;i++){
        double p=beta[0];
        for(int j=0;j<inputCols;j++) p+=beta[j+1]*X[i][j+1];
        y_pred[i]=p;
    }

    double r2 = computeR2(y,y_pred,n);

    cout.setf(ios::fixed); cout.precision(6);
    cout << "Model coefficients (intercept first):\n";
    for(int i=0;i<inputCols+1;i++) cout << beta[i] << (i==inputCols?"\n":" ");
    cout << "R2: " << r2 << "\n";

    if(!saveModel(modelFile,beta,inputCols+1)) cerr<<"Warning: Could not save model\n";
    cout<<"Training finished.\n";

    for(int i=0;i<n;i++) delete[] data[i]; delete[] data;
    for(int i=0;i<n;i++) delete[] X[i]; delete[] X;
    delete[] y; delete[] beta; delete[] y_pred;

    return 0;
}
