#include<opencv2/opencv.hpp>
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
using namespace cv;
using namespace std;

int main()
{
/////////////////////////////////////////////////////////////////////
/////////////////////////////layer 1 conv1///////////////////////////
    Mat image;
    image = imread("test7.jpg",IMREAD_GRAYSCALE);
    float input_image[1][28][28] = {0};

    for(int k=0; k<1; k++)
    {
        for(int i=0; i<28; i++)
        {
            for(int j=0; j<28; j++)
            {
                uchar* pixel = image.ptr<uchar>(i);
                input_image[k][i][j] = pixel[j];
            }
        }
    }

    for(int k=0; k<1; k++)
    {
        for(int i=0;i<28;i++)
        {
            for(int j=0;j<28;j++)
            {
                printf(" %.2f ",input_image[k][i][j]);
            }
            printf("\n");
        }
    }

    FILE *fp_conv1;
    fp_conv1=fopen("/home/socmgr/etri/0729/lenet_conv1_weight.txt","r");

    float ****conv1_filter = NULL; //convolution filter

    float conv1_pad_result[1][28][28] = {0};
    float conv1_bias[20][1][1][1] = {0};
    float conv1_result[20][24][24] = {0};

    int conv1_W_row = 28;
    int conv1_W_col = 28;
    int conv1_C = 1;
    int conv1_P = 0;
    int conv1_S = 1;
    int conv1_F_num = 20;
    int conv1_F = 5;

    int conv1_resultsize_row = (conv1_W_row - conv1_F + 2 * conv1_P) / conv1_S + 1;
    int conv1_resultsize_col = (conv1_W_col - conv1_F + 2 * conv1_P) / conv1_S + 1;

//filter malloc
    conv1_filter = (float****)malloc(conv1_F_num*sizeof(float***));
    for(int k=0; k<conv1_F_num; k++)
    {
        conv1_filter[k] = (float***)malloc(conv1_C*sizeof(float**));
        for(int i=0; i<conv1_C; i++)
        {
            conv1_filter[k][i] = (float**)malloc(conv1_F*sizeof(float*));
            for(int j=0; j<conv1_F; j++)
            {
                conv1_filter[k][i][j] = (float*)malloc(conv1_F*sizeof(float));
            }
        }
    }

//filter value
    float temp=0;
    for(int i=0;i<20;i++){
        for(int j=0;j<1;j++){
            for(int k=0;k<5;k++){
                for(int l=0;l<5;l++){
                    fscanf(fp_conv1,"%f",&temp);
                    conv1_filter[i][j][k][l]=temp;
                }
            }
        }
    }

    fclose(fp_conv1);

//conv1 padding
    for(int k=0;k<conv1_C;k++)
    {
        for(int i=0; i<conv1_W_row+2*conv1_P; i++)
        {
            for(int j=0; j<conv1_W_col+2*conv1_P; j++)
            {
                if(i < conv1_P || j<conv1_P || i > conv1_W_row + conv1_P - 1 || j > conv1_W_col + conv1_P - 1){conv1_pad_result[k][i][j] = 0;}
                else{conv1_pad_result[k][i][j] = input_image[k][i-conv1_P][j-conv1_P];}
            }
        }
    }

//////////////convolution 1
   for(int n=0; n<conv1_F_num;n++)
   {
        for(int x=0;x<conv1_resultsize_row;x++)
        {
            for(int y=0;y<conv1_resultsize_col;y++)
            {
                for(int k=0;k<conv1_C;k++)
                {   
                    for(int i=0;i<conv1_F;i++)
                    {
                        for(int j=0;j<conv1_F;j++)
                        {
                            conv1_result[n][x][y] += conv1_pad_result[k][i+x*conv1_S][j+y*conv1_S] * conv1_filter[n][k][i][j];
                        }
                    }
                }
                //conv1_result[n][x][y] += conv1_bias[n][0][0][0];
            }
        }
   }
/*
   for(int i=0;i<20;i++)
    {
        for(int j=0;j<24;j++)
        {
            for(int k=0;k<24;k++)
            {
                printf(" %f ",conv1_result[i][j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
*/

   printf("\nconv1 complete\n");
///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////layer 2 pooling1///////////////////////////////////////////////////
    float pool1_result[20][12][12] = { 0 };

    int pool1_W_row = 24;
    int pool1_W_col = 24;
    int pool1_C = 20;
    int pool1_S = 2;
    int pool1_F = 2;

    int pool1_resultsize_row = (pool1_W_row - pool1_F)/pool1_S + 1;
    int pool1_resultsize_col = (pool1_W_col - pool1_F)/pool1_S + 1;

    float pool1_max;
    for(int k=0;k<pool1_C;k++)
    {
        for(int x=0;x<pool1_resultsize_row;x++)
        {
            for(int y=0;y<pool1_resultsize_col;y++)
            {
                //pool1_max=0;
                pool1_max=conv1_result[k][x*pool1_S][y*pool1_S];
                for(int i=0;i<pool1_F;i++)
                {
                    for(int j=0;j<pool1_F;j++)
                    {
                        if(pool1_max < conv1_result[k][i+x*pool1_S][j+y*pool1_S])
                        {
                            pool1_max = conv1_result[k][i+x*pool1_S][j+y*pool1_S];
                        }
                    }
                }
                pool1_result[k][x][y] = pool1_max;
            }
        }
    }

    for(int i=0;i<20;i++)
    {
        for(int j=0;j<12;j++)
        {
            for(int k=0;k<12;k++)
            {
                printf(" %f ",pool1_result[i][j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    printf("\n layer2 complete\n");

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////layer 3 conv2//////////////////////////////////////
    FILE *fp_conv2;
    fp_conv2=fopen("/home/socmgr/etri/0729/lenet_conv2_weight.txt","r");

    float ****conv2_filter = NULL; //convolution filter

    float conv2_pad_result[20][12][12] = {0};
    float conv2_bias[20][1][1][1] = {0};
    float conv2_result[50][8][8] = {0};

    int conv2_W_row = 12;
    int conv2_W_col = 12;
    int conv2_C = 20;
    int conv2_P = 0;
    int conv2_S = 1;
    int conv2_F_num = 50;
    int conv2_F = 5;

    int conv2_resultsize_row = (conv2_W_row - conv2_F + 2 * conv2_P) / conv2_S + 1;
    int conv2_resultsize_col = (conv2_W_col - conv2_F + 2 * conv2_P) / conv2_S + 1;

//filter malloc
    conv2_filter = (float****)malloc(conv2_F_num*sizeof(float***));
    for(int k=0; k<conv2_F_num; k++)
    {
        conv2_filter[k] = (float***)malloc(conv2_C*sizeof(float**));
        for(int i=0; i<conv2_C; i++)
        {
            conv2_filter[k][i] = (float**)malloc(conv2_F*sizeof(float*));
            for(int j=0; j<conv2_F; j++)
            {
                conv2_filter[k][i][j] = (float*)malloc(conv2_F*sizeof(float));
            }
        }
    }

//filter value
    float temp2=0;
    for(int i=0;i<50;i++){
        for(int j=0;j<20;j++){
            for(int k=0;k<5;k++){
                for(int l=0;l<5;l++){
                    fscanf(fp_conv2,"%f",&temp2);
                    conv2_filter[i][j][k][l]=temp2;
                }
            }
        }
    }

    fclose(fp_conv2);
/*
    for(int i=0;i<50;i++){
        for(int j=0;j<20;j++){
            for(int k=0;k<5;k++){
                for(int l=0;l<5;l++){
                    printf("%f\n",conv2_filter[i][j][k][l]);
                }
            }
        }
    }
    */
//conv2 padding
    for(int k=0;k<conv2_C;k++)
    {
        for(int i=0; i<conv2_W_row+2*conv2_P; i++)
        {
            for(int j=0; j<conv2_W_col+2*conv2_P; j++)
            {
                if(i < conv2_P || j<conv2_P || i > conv2_W_row + conv2_P - 1 || j > conv2_W_col + conv2_P - 1){conv2_pad_result[k][i][j] = 0;}
                else{conv2_pad_result[k][i][j] = pool1_result[k][i-conv2_P][j-conv2_P];}
            }
        }
    }

    for(int k=0;k<conv2_C;k++)
    {
        for(int i=0; i<conv2_W_row+2*conv2_P; i++)
        {
            for(int j=0; j<conv2_W_col+2*conv2_P; j++)
            {
                printf("%f",conv2_pad_result[k][i][j]);
            }
        }
    }

//////////////convolution 2
   for(int n=0; n<conv2_F_num;n++)
   {
        for(int x=0;x<conv2_resultsize_row;x++)
        {
            for(int y=0;y<conv2_resultsize_col;y++)
            {
                for(int k=0;k<conv2_C;k++)
                {   
                    for(int i=0;i<conv2_F;i++)
                    {
                        for(int j=0;j<conv2_F;j++)
                        {
                            conv2_result[n][x][y] += conv2_pad_result[k][i+x*conv2_S][j+y*conv2_S] * conv2_filter[n][k][i][j];
                        }
                    }
                }
               // conv2_result[n][x][y] += conv2_bias[n][0][0][0];
            }
        }
   }

   for(int i=0;i<50;i++)
    {
        for(int j=0;j<8;j++)
        {
            for(int k=0;k<8;k++)
            {
                printf(" %f ",conv2_result[i][j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

   printf("\nconv2 complete\n");
///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////layer 4 pooling2///////////////////////////////////////////////////
    float pool2_result[50][4][4] = { 0 };

    int pool2_W_row = 8;
    int pool2_W_col = 8;
    int pool2_C = 50;
    int pool2_S = 2;
    int pool2_F = 2;

    int pool2_resultsize_row = (pool2_W_row - pool2_F)/pool2_S + 1;
    int pool2_resultsize_col = (pool2_W_col - pool2_F)/pool2_S + 1;

    float pool2_max;
    for(int k=0;k<pool2_C;k++)
    {
        for(int x=0;x<pool2_resultsize_row;x++)
        {
            for(int y=0;y<pool2_resultsize_col;y++)
            {
                pool2_max=conv2_result[k][x*pool2_S][y*pool2_S];
                for(int i=0;i<pool2_F;i++)
                {
                    for(int j=0;j<pool2_F;j++)
                    {
                        if(pool2_max < conv2_result[k][i+x*pool2_S][j+y*pool2_S])
                        {
                            pool2_max = conv2_result[k][i+x*pool2_S][j+y*pool2_S];
                        }
                    }
                }
                pool2_result[k][x][y] = pool2_max;
            }
        }
    }
/*
    for(int i=0;i<50;i++)
    {
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            {
                printf(" %f ",pool2_result[i][j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
*/
    printf("\n layer4 complete\n");
///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////layer 5 ip1///////////////////////////////////////////////////

    FILE *fp_ip1;
    fp_ip1=fopen("/home/socmgr/etri/0729/lenet_ip1_weight.txt","r");

    int ip1_F_num = 500;
    int ip1_F = 4;
    int ip1_C = 50;

    int ip1_resultsize = ip1_C * ip1_F * ip1_F;

    float ****ip1_filter = NULL;
    
    float ip1_result[500] = {0};

//ip1 filter malloc
    ip1_filter = (float****)malloc(ip1_F_num*sizeof(float***));
    for(int k=0; k<ip1_F_num; k++)
    {
        ip1_filter[k] = (float***)malloc(ip1_C*sizeof(float**));
        for(int i=0; i<ip1_C; i++)
        {
            ip1_filter[k][i] = (float**)malloc(ip1_F*sizeof(float*));
            for(int j=0; j<ip1_F; j++)
            {
                ip1_filter[k][i][j] = (float*)malloc(ip1_F*sizeof(float));
            }
        }
    }

//ip1 filter malloc
    float temp3=0;
    for(int i=0; i<500; i++)
    {
        for(int j=0; j<50; j++)
        {
            for(int k=0; k<4; k++)
            {
                for(int l=0; l<4; l++)
                {
                    fscanf(fp_ip1,"%f",&temp3);
                    ip1_filter[i][j][k][l]=temp3;
                }
            }
        }
    }
    fclose(fp_ip1);

    for(int n=0; n<ip1_F_num;n++)
    {
        for(int k=0; k<ip1_C;k++)
        {
            for(int i=0; i<ip1_F;i++)
            {
                for(int j=0; j<ip1_F;j++)
                {
                    ip1_result[n] += ip1_filter[n][k][i][j] * pool2_result[k][i][j];
                }
            }
        }
    }
    /*
    int count = 0;
    for(int i=0; i<ip1_F_num;i++)
    {
        count= count+1;
        printf("%d : %f\n",count, ip1_result[i]);
    }
    */
    

    printf("\nlayer5 ip1 complete\n");
///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// ReLU///////////////////////////////////////////////////
    for(int i=0; i<ip1_F_num;i++)
    {
        if(ip1_result[i]<0){ ip1_result[i] = 0;}
        else {ip1_result[i] = ip1_result[i];}
    }

/*
    int count2 = 0;
    for(int i=0; i<ip1_F_num;i++)
    {
        count2= count2+1;
        printf("%d : %f\n",count2, ip1_result[i]);
    }
    */
    printf("\nReLU complete\n\n");
///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////layer 6 ip2///////////////////////////////////////////////////
    FILE *fp_ip2;
    fp_ip2=fopen("/home/socmgr/etri/0729/lenet_ip2_weight.txt","r");

    int ip2_F_num = 10;
    int ip2_F = 1;
    int ip2_C = 500;

    int ip2_resultsize = ip2_C * ip2_F * ip2_F;

    float ****ip2_filter = NULL;
    
    float ip2_result[10] = {0};

//ip2 filter malloc
    ip2_filter = (float****)malloc(ip2_F_num*sizeof(float***));
    for(int k=0; k<ip2_F_num; k++)
    {
        ip2_filter[k] = (float***)malloc(ip2_C*sizeof(float**));
        for(int i=0; i<ip2_C; i++)
        {
            ip2_filter[k][i] = (float**)malloc(ip2_F*sizeof(float*));
            for(int j=0; j<ip2_F; j++)
            {
                ip2_filter[k][i][j] = (float*)malloc(ip2_F*sizeof(float));
            }
        }
    }

//ip2 filter value
    float temp4=0;
    for(int i=0; i<10; i++)
    {
        for(int j=0; j<500; j++)
        {
            for(int k=0; k<1; k++)
            {
                for(int l=0; l<1; l++)
                {
                    fscanf(fp_ip2,"%f",&temp4);
                    ip2_filter[i][j][k][l]=temp4;
                }
            }
        }
    }
    fclose(fp_ip2);

    for(int n=0; n<ip2_F_num;n++)
    {
        for(int k=0; k<ip2_C;k++)
        {
            for(int i=0; i<ip2_F;i++)
            {
                for(int j=0; j<ip2_F;j++)
                {
                    ip2_result[n] += ip2_filter[n][k][i][j] * ip1_result[k];
                }
            }
        }
    }

    int count3 = 0;
    for(int i=0; i<ip2_F_num;i++)
    {
        printf("%d : %f\n",count3, ip2_result[i]);
        count3= count3+1;
    }
    

    printf("\nlayer6 ip2 complete\n");

//softmax
    int num=0;
    float softmax = 0;
    float sum_exp;
    float softmax_result[10] = {0};
    for(int i = 0; i < ip2_F_num;i++)
    {
        if(softmax < ip2_result[i]){softmax = ip2_result[i];} //softmax는 최댓값
    }

    for(int i = 0; i < ip2_F_num;i++)
    {
        sum_exp += exp(ip2_result[i] - softmax);
    }

    for(int i = 0; i < ip2_F_num;i++)
    {
        softmax_result[i] = exp(ip2_result[i] - softmax)/sum_exp ;
    }

    for(int i=0; i<ip2_F_num; i++)
    {
        printf(" %d : %f\n",num,softmax_result[i]);
        num+=1;
    }


//free
    for(int i=0;i<conv1_F_num;i++)
    {
        for(int j=0;j<conv1_C;j++)
        {
            for(int k=0; k<conv1_F;k++)
            {
                free(*(*(*(conv1_filter+i)+j)+k));
            }
            free(*(*(conv1_filter+i)+j));
        }
        free(*(conv1_filter+i));
    }
    free(conv1_filter);

    for(int i=0;i<conv2_F_num;i++)
    {
        for(int j=0;j<conv2_C;j++)
        {
            for(int k=0; k<conv2_F;k++)
            {
                free(*(*(*(conv2_filter+i)+j)+k));
            }
            free(*(*(conv2_filter+i)+j));
        }
        free(*(conv2_filter+i));
    }
    free(conv2_filter);

    for(int i=0;i<ip1_F_num;i++)
    {
        for(int j=0;j<ip1_C;j++)
        {
            for(int k=0; k<ip1_F;k++)
            {
                free(*(*(*(ip1_filter+i)+j)+k));
            }
            free(*(*(ip1_filter+i)+j));
        }
        free(*(ip1_filter+i));
    }
    free(ip1_filter);

    for(int i=0;i<ip2_F_num;i++)
    {
        for(int j=0;j<ip2_C;j++)
        {
            for(int k=0; k<ip2_F;k++)
            {
                free(*(*(*(ip2_filter+i)+j)+k));
            }
            free(*(*(ip2_filter+i)+j));
        }
        free(*(ip2_filter+i));
    }
    free(ip2_filter);


}