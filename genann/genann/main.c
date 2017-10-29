/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015, 2016 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "genann.h"
#include "minctest.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "example1.h"
#include "example2.h"
#include "example3.h"
#include "example4.h"
#include "example4dng.h"
#include "example5dng.h"



void basic() {
    genann *ann = genann_init(1, 0, 0, 1);

    lequal(ann->total_weights, 2);
    double a;


    a = 0;
    ann->weight[0] = 0;
    ann->weight[1] = 0;
    lfequal(0.5, *genann_run(ann, &a));

    a = 1;
    lfequal(0.5, *genann_run(ann, &a));

    a = 11;
    lfequal(0.5, *genann_run(ann, &a));

    a = 1;
    ann->weight[0] = 1;
    ann->weight[1] = 1;
    lfequal(0.5, *genann_run(ann, &a));

    a = 10;
    ann->weight[0] = 1;
    ann->weight[1] = 1;
    lfequal(1.0, *genann_run(ann, &a));

    a = -10;
    lfequal(0.0, *genann_run(ann, &a));

    genann_free(ann);
}


void xor() {
    genann *ann = genann_init(2, 1, 2, 1);
    ann->activation_hidden = genann_act_threshold;
    ann->activation_output = genann_act_threshold;

    lequal(ann->total_weights, 9);

    /* First hidden. */
    ann->weight[0] = .5;
    ann->weight[1] = 1;
    ann->weight[2] = 1;

    /* Second hidden. */
    ann->weight[3] = 1;
    ann->weight[4] = 1;
    ann->weight[5] = 1;

    /* Output. */
    ann->weight[6] = .5;
    ann->weight[7] = 1;
    ann->weight[8] = -1;


    double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double output[4] = {0, 1, 1, 0};

    lfequal(output[0], *genann_run(ann, input[0]));
    lfequal(output[1], *genann_run(ann, input[1]));
    lfequal(output[2], *genann_run(ann, input[2]));
    lfequal(output[3], *genann_run(ann, input[3]));

    genann_free(ann);
}


void backprop() {
    genann *ann = genann_init(1, 0, 0, 1);

    double input, output;
    input = .5;
    output = 1;

    double first_try = *genann_run(ann, &input);
    genann_train(ann, &input, &output, .5);
    double second_try = *genann_run(ann, &input);
    lok(fabs(first_try - output) > fabs(second_try - output));

    genann_free(ann);
}


void train_and() {
    double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double output[4] = {0, 0, 0, 1};

    genann *ann = genann_init(2, 0, 0, 1);

    int i, j;

    for (i = 0; i < 50; ++i) {
        for (j = 0; j < 4; ++j) {
            genann_train(ann, input[j], output + j, .8);
        }
    }

    ann->activation_output = genann_act_threshold;
    lfequal(output[0], *genann_run(ann, input[0]));
    lfequal(output[1], *genann_run(ann, input[1]));
    lfequal(output[2], *genann_run(ann, input[2]));
    lfequal(output[3], *genann_run(ann, input[3]));

    genann_free(ann);
}


void train_or() {
    double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double output[4] = {0, 1, 1, 1};

    genann *ann = genann_init(2, 0, 0, 1);
    genann_randomize(ann);

    int i, j;

    for (i = 0; i < 50; ++i) {
        for (j = 0; j < 4; ++j) {
            genann_train(ann, input[j], output + j, .8);
        }
    }

    ann->activation_output = genann_act_threshold;
    lfequal(output[0], *genann_run(ann, input[0]));
    lfequal(output[1], *genann_run(ann, input[1]));
    lfequal(output[2], *genann_run(ann, input[2]));
    lfequal(output[3], *genann_run(ann, input[3]));

    genann_free(ann);
}



void train_xor() {
    double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double output[4] = {0, 1, 1, 0};

    genann *ann = genann_init(2, 1, 2, 1);

    int i, j;

    for (i = 0; i < 500; ++i) {
        for (j = 0; j < 4; ++j) {
            genann_train(ann, input[j], output + j, 3);
        }
        /* printf("%1.2f ", xor_score(ann)); */
    }

    ann->activation_output = genann_act_threshold;
    lfequal(output[0], *genann_run(ann, input[0]));
    lfequal(output[1], *genann_run(ann, input[1]));
    lfequal(output[2], *genann_run(ann, input[2]));
    lfequal(output[3], *genann_run(ann, input[3]));

    genann_free(ann);
}



void persist() {
    genann *first = genann_init(1000, 5, 50, 10);

    FILE *out = fopen("/Users/blue5/Documents/GitHub-blue5/genann/genann/genann/files/persist.txt", "w");
    genann_write(first, out);
    fclose(out);


    FILE *in = fopen("/Users/blue5/Documents/GitHub-blue5/genann/genann/genann/files/persist.txt", "r");
    genann *second = genann_read(in);
    fclose(out);

    lequal(first->inputs, second->inputs);
    lequal(first->hidden_layers, second->hidden_layers);
    lequal(first->hidden, second->hidden);
    lequal(first->outputs, second->outputs);
    lequal(first->total_weights, second->total_weights);

    int i;
    for (i = 0; i < first->total_weights; ++i) {
        lok(first->weight[i] == second->weight[i]);
    }

    genann_free(first);
    genann_free(second);
}


void copy() {
    genann *first = genann_init(1000, 5, 50, 10);

    genann *second = genann_copy(first);

    lequal(first->inputs, second->inputs);
    lequal(first->hidden_layers, second->hidden_layers);
    lequal(first->hidden, second->hidden);
    lequal(first->outputs, second->outputs);
    lequal(first->total_weights, second->total_weights);

    int i;
    for (i = 0; i < first->total_weights; ++i) {
        lfequal(first->weight[i], second->weight[i]);
    }

    genann_free(first);
    genann_free(second);
}


void sigmoid() {
    double i = -20;
    const double max = 20;
    const double d = .0001;

    while (i < max) {
        lfequal(genann_act_sigmoid(i), genann_act_sigmoid_cached(i));
        i += d;
    }
}


int main(int argc, char *argv[])
{
    printf("GENANN okok SUITE\n");

    srand(100);
/*
    lrun("basic", basic);
    lrun("xor", xor);
    lrun("backprop", backprop);
    lrun("train and", train_and);
    lrun("train or", train_or);
    lrun("train xor", train_xor);
    lrun("persist", persist);
    lrun("copy", copy);
    lrun("sigmoid", sigmoid);

    lresults();

    printf("GENANN okok results -> %i \n\n", lfails);

    printf("\nexample 1 results -> %i \n\n", example1());
    printf("\nexample 2 results -> %i \n\n", example2());
    
    FILE *fp;
    fp=fopen("/Users/blue5/Documents/GitHub-blue5/genann/genann/genann/files/okoka2j.bin", "wb");
    char x[10]="ABCDEFGHIJ";
    fwrite(x, sizeof(x[0]), sizeof(x)/sizeof(x[0]), fp);
    fclose(fp);
    
    FILE *fp4;
    char buff4[255];
    
    fp4 = fopen("/Users/blue5/Documents/GitHub-blue5/genann/genann/genann/files/okoka2j.bin", "r");
    fscanf(fp4, "%s", buff4);
    printf("4a : %s\n", buff4 );
    
    //fgets(buff4, 255, (FILE*)fp4);
    //printf("2: %s\n", buff4 );
    
    //fgets(buff4, 255, (FILE*)fp4);
    //printf("3: %s\n", buff4 );
    fclose(fp4);

    FILE *fp2;
    
    fp2 = fopen("/Users/blue5/Documents/GitHub-blue5/genann/genann/genann/files/okokb2m.txt", "w+"); // Ok to read and write but it goes to DEBUG !!!!!
       // something like /Users/blue5/Library/Developer/Xcode/DerivedData/genann-eulbjjefxxriwefgbidjfwbcykdu/Build/Products/Debug
    fprintf(fp2, "okokb2m\n This is okoking for fprintf...\n");
    fputs("This is okoking for fputs...\n", fp2);
    fclose(fp2);
    
    FILE *fp3;
    char buff[255];
    
    fp3 = fopen("/Users/blue5/Documents/GitHub-blue5/genann/genann/genann/files/okokb2m.txt", "r");
    fscanf(fp3, "%s", buff);
    printf("3a : %s\n", buff );
    
    fgets(buff, 255, (FILE*)fp3);
    printf("3b: %s\n", buff );
    
    fgets(buff, 255, (FILE*)fp3);
    printf("3c: %s\n", buff );
    fclose(fp3);
    
    printf("\nexample 3 results -> %i \n\n", example3());
    printf("\nexample 4 results -> %i \n\n", example4());
 
    printf("\nexample 5dng results -> %i \n\n", example5dng());
 
*/
    
    printf("\nexample 4 dng results -> %i \n\n", example4dng());
    
    return lfails != 0;
}
