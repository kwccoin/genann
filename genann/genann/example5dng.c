#include <stdio.h>
#include "genann.h"

int example5dng() //int argc, char *argv[])
{
    printf("GENANN example 5dng.\n");
    printf("Train a small ANN to the XY function using backpropagation but not working well.\n\n");

    /* Input and expected out data for the XOR function. */
    const double input[12][2] = {{1, 2}, {-1, 2}, {1, -2}, {-1, -2},
                                 {0, 0},  {0, 1},  {1, 0},   {1, 1},
                                 {2, 3}, {-2, 3}, {2, -3}, {-2, -3}};
    const double output[12]   = {    2,      -2,      -2,        2,
                                     0,       0,       0,        1,
                                     6,      -6,      -6,        6};
    int i;

    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */
    genann *ann = genann_init(2, 1, 3, 1); // 2 input and 1 hidden layer with 2 neuron and 1 output
     // change to 2,2,8,1 still not ok

    /* Train on the four labeled data points many times. */
    for (i = 0; i < 30000; ++i) {  // change from 300 to 3000 to 300000 not help
        genann_train(ann, input[0],  output + 0,  3); // 3 to 0.3 not help
        genann_train(ann, input[1],  output + 1,  3); // try 30 not ok and the first 4 predict wrongly
        genann_train(ann, input[2],  output + 2,  3);
        genann_train(ann, input[3],  output + 3,  3);
        genann_train(ann, input[4],  output + 4,  3); // need to add training !!!
        genann_train(ann, input[5],  output + 5,  3);
        genann_train(ann, input[6],  output + 6,  3);
        genann_train(ann, input[7],  output + 7,  3);
        genann_train(ann, input[8],  output + 8,  3); // need to add training !!!
        genann_train(ann, input[9],  output + 9,  3);
        genann_train(ann, input[10], output + 10, 3);
        genann_train(ann, input[11], output + 11, 3);
    }

    /* Run the network and see what it predicts. */
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[0][0], input[0][1], *genann_run(ann, input[0]));
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[1][0], input[1][1], *genann_run(ann, input[1]));
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[2][0], input[2][1], *genann_run(ann, input[2]));
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[3][0], input[3][1], *genann_run(ann, input[3]));
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[4][0], input[4][1], *genann_run(ann, input[4]));
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[5][0], input[5][1], *genann_run(ann, input[5]));
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[6][0], input[6][1], *genann_run(ann, input[6]));
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[7][0], input[7][1], *genann_run(ann, input[7]));
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[8][0], input[8][1], *genann_run(ann, input[8]));
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[9][0], input[9][1], *genann_run(ann, input[9]));
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[10][0], input[10][1], *genann_run(ann, input[10]));
    printf("example5dng - Output for [%1.f, %1.f] is %1.f.\n", input[11][0], input[11][1], *genann_run(ann, input[11]));

    FILE *fp5w;
    fp5w = fopen("/Users/blue5/Documents/GitHub-blue5/genann/genann/genann/files/examples5dng1.txt", "w+");
    genann_write(ann, fp5w);
    fclose(fp5w);
    
    genann *ann2;
    
    FILE *fp5r;
    fp5r = fopen("/Users/blue5/Documents/GitHub-blue5/genann/genann/genann/files/examples5dng1.txt", "r");
    ann2 = genann_read(fp5r);
    fclose(fp5r);
    
    printf("\n\n === readio === \n\n");

    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[0][0],  input[0][1],  *genann_run(ann2, input[0]));
    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[1][0],  input[1][1],  *genann_run(ann2, input[1]));
    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[2][0],  input[2][1],  *genann_run(ann2, input[2]));
    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[3][0],  input[3][1],  *genann_run(ann2, input[3]));
    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[4][0],  input[4][1],  *genann_run(ann2, input[4]));
    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[5][0],  input[5][1],  *genann_run(ann2, input[5]));
    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[6][0],  input[6][1],  *genann_run(ann2, input[6]));
    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[7][0],  input[7][1],  *genann_run(ann2, input[7]));
    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[8][0],  input[8][1],  *genann_run(ann2, input[8]));
    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[9][0],  input[9][1],  *genann_run(ann2, input[9]));
    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[10][0], input[10][1], *genann_run(ann2, input[10]));
    printf("exradio5dng - Output for [%1.f, %1.f] is %1.f.\n", input[11][0], input[11][1], *genann_run(ann2, input[11]));

    
    genann_free(ann);
    return 0;
}
