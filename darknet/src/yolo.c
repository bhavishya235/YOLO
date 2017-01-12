#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include <dirent.h>
#include <string.h>

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

// ********** EDITED *************
typedef struct{
    float left, top, right, bot, score;
} boxPerson;

typedef struct {
    char imageName[256];
    int numBbox;
    boxPerson eachBox[100];
} record;


// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
int get_all_files_names_within_folder(char* dirname, char* ext, char* ret[]){
    char* str;
    DIR *pDIR;
    struct dirent *entry;
    int i = 0;
    if( (pDIR=opendir(dirname)) != NULL ){
        while(entry = readdir(pDIR)){
            if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ){
                str = entry->d_name;
                if(strstr(str, ext) != NULL) {
                    ret[i] = (char*)malloc(strlen(str)+1);
                    strcpy(ret[i++],str);
                    // printf("%s\n", ret[i-1]);
                }
            }
        }
        closedir(pDIR);
    }
    return i;
}

void get_record(image im, int num, float thresh, box *boxes, float **probs, char **names, int classes, record *r)
{
    int i = 0, j = 0;
    for(; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(strcmp(names[class],"person") == 0 && prob > thresh){
            if(j>=100) fprintf(stderr, "Number of people in an image exceeded maximum limit of 100\n");
            // int width = im.h * .012;
            // int offset = class*1 % classes;
            box b = boxes[i];

            float left  = b.x-b.w/2.;
            float right = b.x+b.w/2.;
            float top   = b.y-b.h/2.;
            float bot   = b.y+b.h/2.;

            if(left < 0.0) left = 0.0;
            if(right > 1.0) right = 1.0;
            if(top < 0.0) top = 0.0;
            if(bot > 1.0) bot = 1.0;

            r->eachBox[j].left = left;
            r->eachBox[j].right = right;
            r->eachBox[j].top = top;
            r->eachBox[j].bot = bot;
            r->eachBox[j].score = prob;
            j+=1;
        }
    }
    r->numBbox = j;
}

void recordToString(record *r, char *r_str)
{
    strcpy(r_str, r->imageName);
    strncat(r_str, ";", 2);
    char s[11];
    sprintf(s,"%d", r->numBbox);
    strncat(r_str, s, 11);strncat(r_str, ";", 2);
    int i;
    for(i = 0; i< r->numBbox;i++) {
        // left, top, right, bot, score in % of image h and w
        strcpy(s, "");sprintf(s,"%f", r->eachBox[i].left);
        strncat(r_str, s, 11);strncat(r_str, ",", 2);
        
        strcpy(s, "");sprintf(s,"%f", r->eachBox[i].top);
        strncat(r_str, s, 11);strncat(r_str, ",", 2);
        
        strcpy(s, "");sprintf(s,"%f", r->eachBox[i].right);
        strncat(r_str, s, 11);strncat(r_str, ",", 2);
        
        strcpy(s, "");sprintf(s,"%f", r->eachBox[i].bot);
        strncat(r_str, s, 11);strncat(r_str, ",", 2);
        
        strcpy(s, "");sprintf(s,"%f", r->eachBox[i].score);
        strncat(r_str, s, 11);
        
        if (i!=r->numBbox-1) strncat(r_str, ";", 2);
    }
}
// **************************************************************************

void train_yolo(char *cfgfile, char *weightfile)
{
    char *train_images = "/data/voc/train.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_yolo(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            get_detection_boxes(l, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, boxes, probs, l.side*l.side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_yolo_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = 0;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        get_detection_boxes(l, orig.w, orig.h, thresh, probs, boxes, 1);
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{
    image *alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, 20);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, 20);
        save_image(im, "predictions");
        show_image(im, "predictions");

        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}


// ********** EDITED *************
void test_yolo_custom(char *cfgfile, char *weightfile, char *dirname, float thresh, char *resultFolder, char *resultImgFolder)
{
    image *alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    printf("n = %d\n", l.n);
    char* fileNames[50000];
    int nFiles = 0;
    if(dirname){
        nFiles = get_all_files_names_within_folder(dirname, ".jpg", fileNames);
    } 
    else {
        printf("Directory name not provided");
        return;
    }
    printf("no. of files = %d\n", nFiles);
    int i;
    for(i=0; i < nFiles; i++){
        char *currFile = fileNames[i];
        strncpy(input, dirname, 256);
        printf("%d processed\n", i);
        if (((i+1)%100)==0) printf("%d processed\n", i);
        //printf("File Processing: %d %s\n", i,currFile);
        strcat(input, currFile);
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, 20);

    // saving record
        record r;
        char r_str[1024];
        strcpy(r.imageName, input);
        get_record(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 20, &r);
        recordToString(&r, r_str);	// converting record to string format
        // printf("%s\n", r_str);
        // printf("Saving result in file %s\n", resultFolder);
        FILE *fp;
        char saveFile[256];
        strncpy(saveFile, resultFolder, 256);
        char mystr[256] = "";
        strncpy(mystr,currFile,(int)(strchr(currFile,'.')-currFile));
        // printf("saving in : %s \n",saveFile);
        strcat(mystr,".txt");
        strcat(saveFile, mystr);
        //printf("saving in : %s \n",saveFile);
        fp = fopen(saveFile, "w+");
        fprintf(fp, "%s\n", r_str);
        fclose(fp);

    // Saving image
        char saveImgName[256];
        strncpy(saveImgName,resultImgFolder, 256);
        strcat(saveImgName, currFile);
        save_image(im, saveImgName);
        // show_image(im, "predictions");

        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
    }
}
// **************************************************************************

void run_yolo(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        // ./darknet yolo test cfg/yolo.cfg yolo.weights data/dog.jpg [resultFile] 
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    char *resultFolder = (argc > 6) ? argv[6]: 0;
    char *resultImgFolder = (argc > 7) ? argv[7]: 0;
    if(0==strcmp(argv[2], "test_custom")) {
        if (resultFolder && resultImgFolder) test_yolo_custom(cfg, weights, filename, thresh, resultFolder, resultImgFolder);
        else fprintf(stderr, "usage: %s %s test_custom [cfg] [weights (optional)] [dirname] [resultFolder] [resultImgFolder]\n", argv[0], argv[1]);
    }
    else if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, 20, frame_skip, prefix);
}
