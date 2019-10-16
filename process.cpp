 #include "process.h"

void* process(void*) {
    
    sl::Mat depths;
    sl::Mat normals;
    // sl::float4 floor_normals = sl::float4(sl::Vector4<float>(0, 0, 0, 0));
    
    // Size in pixels of the previously detected object
    int previous_c = 0;
    
    // Array position of the previously detected object
    int previous_row = -1;
    int previous_col = -1;

    //std::vector<std::vector<int> > object_pixels(NUM_ROWS, std::vector<int> (NUM_BLOCKS));
    //std::vector<std::vector<std::vector<float>>> object_pixels(NUM_ROWS, std::vector<std::vector<float>>(NUM_BLOCKS, std::vector<float>(NUM_BLOCKS,2)));
    std::vector<std::vector<int> > object_pixels(NUM_ROWS, std::vector<int> (NUM_BLOCKS));
    std::vector<std::vector<int> > min_object_pixels(NUM_ROWS, std::vector<int> (NUM_BLOCKS));
    std::vector<std::vector<float> > dist_object(NUM_ROWS, std::vector<float> (NUM_BLOCKS));
	
    float prev_depth_value = 0;
    int xend;

    // local version of shared variables
    float max_dist;
    float min_dist;
    int threshold;
    cv::Mat pixels;

    sl::Mat d;
    sl::Mat n;

    
    // for debugging
    int num_frames = 1;
    std::ofstream file;
    if (DEBUG)
        file.open("./debug/log.txt");
    
    int skipped_frames = 0; // used to delay arrow update and avoid vibrations
        //min_obj initialization
    for (int r = 0; r < NUM_ROWS; r++) {
            for (int b = 0; b < NUM_BLOCKS; b++) {
                min_object_pixels[r][b] = detection_threshold;
		if (b == 0 || b == NUM_BLOCKS - 1) min_object_pixels[r][b] = min_object_pixels[r][b] * lateral_threshold;
            }
        }
	
    while (get_key() != 'q') {
        /////////// Step 2c: Image Processing ///////////////
        int c_max = 0;
        int row_max = -1;
        int col_max = -1;
	float dis = 0;
        /////////// Step 2c.1: Wait for new frame ///////////////
        // if (get_key() == 'q') break;
        mysem_wait(&sem);
        assert(pthread_mutex_lock(&zed_mutex) == 0);
        zed.retrieveMeasure(depths, sl::MEASURE_DEPTH);
        depths.copyTo(d);
        zed.retrieveMeasure(normals, sl::MEASURE_NORMALS);
        normals.copyTo(n);
        assert(pthread_mutex_unlock(&zed_mutex) == 0);

        assert(pthread_mutex_lock(&min_dist_mutex) == 0);
        min_dist = min_distance;
        assert(pthread_mutex_unlock(&min_dist_mutex) == 0);

        assert(pthread_mutex_lock(&max_dist_mutex) == 0);
        max_dist = max_distance;
        assert(pthread_mutex_unlock(&max_dist_mutex) == 0);

        assert(pthread_mutex_lock(&threshold_mutex) == 0);
        threshold = detection_threshold;
        assert(pthread_mutex_unlock(&threshold_mutex) == 0);
        
        // Safe x_step reset
        assert(pthread_mutex_lock(&step_mutex) == 0);
        x_step = -1;
        assert(pthread_mutex_unlock(&step_mutex) == 0);

        pixels = cv::Mat::zeros(image_size.height, image_size.width, 24);

        /////////// Step 2c.2: Calculate normals ///////////////
        assert(pthread_mutex_lock(&calibrate_mutex) == 0);
        int cal = calibrate_flag;
        calibrate_flag = false;
        assert(pthread_mutex_unlock(&calibrate_mutex) == 0);
        /*if (cal) {
            floor_normals = calibrate(n);
        } */
        
        // Detect objects in current frame
        detection(pixels, n, d, object_pixels,dist_object);
        
        for (int j = 0; j < NUM_ROWS; j++) {
            for (int k = 0; k < NUM_BLOCKS; k++) {
                /*// Increase threshold in lateral columns
                if (k == 0 || k == NUM_BLOCKS - 1) object_pixels[j][k] = object_pixels[j][k] * lateral_threshold;
                // Save biggest object information
                if (object_pixels[j][k] >= threshold and object_pixels[j][k] > c_max) {
                    c_max = object_pixels[j][k];
                    row_max = j;
                    col_max = k;
                }*/
		if (object_pixels[j][k] >= min_object_pixels[j][k] and dist_object[j][k] >= dis){
			dis = dist_object[j][k];
			c_max = object_pixels[j][k];
			row_max = j;
			col_max = k;
		}
            }
        }
            
        // copy detection to shared pixels
        pixeles[pixelesgrab] = pixels;
        updatePixelesGrab();

        // Make the arrow point to the object
        // and avoid vibration of the arrow
        
        
        if ((abs(c_max - previous_c) < threshold) || (skipped_frames <= FRAMES_TO_SKIP)) {
            if (c_max == 0) {
                row_max = -1;
                col_max = -1;
            } else {
                row_max = previous_row;
                col_max = previous_col;
            }
            skipped_frames++;
        } else {
            skipped_frames = 0; // reset skipped frames count
        }
        
        if (DEBUG) {
            file << num_frames << std::endl;
            cv::imwrite("./debug/" + std::to_string(num_frames) + ".jpg", pixels);
            for (int j = 0; j < NUM_ROWS; j++) {
                file << object_pixels[j][0] << " " << object_pixels[j][1] << " " << object_pixels[j][2] << std::endl;
            }
            num_frames++;
            file << std::endl;
        }

        // safe assignment of x_end
        assert(pthread_mutex_lock(&object_mutex) == 0);
        object_row = row_max;
        object_col = col_max;
        assert(pthread_mutex_unlock(&object_mutex) == 0);

        previous_c = c_max;
        previous_row = row_max;
        previous_col = col_max;
        /////////// END Step 2b: Image Processing ///////////////
    }
    if (DEBUG)
        file.close();
    pthread_exit(NULL);
}

void detection(cv::Mat& pixels, sl::Mat& normals, sl::Mat& depths,std::vector<std::vector<int>>& object_pixels,std::vector<std::vector<float>>& dist_object) {
    
    cv::Mat blue_channel;
    cv::Mat green_channel;
    cv::Mat red_channel;
    
    // Convert ZED Mat to OpenCV Mat
    cv::Mat normals_ocv = slMat2cvMat(normals);
    cv::Mat depths_ocv = slMat2cvMat(depths);

    // Remove invalid depth values
    normalizeMat(depths_ocv);

    // Scale depth map to 256 values
    depths_ocv.convertTo(depths_ocv, CV_8U, 255);

    // Extract X normal map (Walls information)
    cv::extractChannel(normals_ocv, blue_channel, 0);

    // Invert X map values to detect walls on the right side of the frame
    cv::Mat blue_inv = blue_channel * -1.0;
    blue_channel.convertTo(blue_channel, CV_8UC3, 255);
    blue_inv.convertTo(blue_inv, CV_8UC3, 255);
    blue_channel = blue_channel + blue_inv;
    //cv::imshow("Normals X", blue_channel);

    // Extract Y normal map (Floor information)
    cv::extractChannel(normals_ocv, green_channel, 1);
    green_channel.convertTo(green_channel, CV_8UC3, 255);

    //depths_ocv = depths_ocv - green_channel;
    //cv::imshow("Depth", depths_ocv);
    //cv::imshow("Original", pixels);

    //cv::imshow("Normals Y", green_channel);
    
    // Extract Z normal map (Obstacles information)
    cv::extractChannel(normals_ocv, red_channel, 2);
    red_channel.convertTo(red_channel, CV_8UC3, 255);
    //cv::imshow("Normals Z", red_channel);

    // Subtract floor and walls pixels in Z values (Obstacle normals)
    red_channel = red_channel - green_channel;
    red_channel = red_channel - blue_channel;
    //cv::imshow("Subs Z", red_channel);


    ////////////////////////////////////////////// Adaptive Thresholding ////////////////////////////////////////////// 

    cv::Mat orig_depths = depths_ocv.clone();
    int ver_center = depths_ocv.rows / 2;
    int step = ver_center / 10;
    int thres = 130;
    cv::Mat roi = depths_ocv(cv::Rect(0, 0, depths_ocv.cols, ver_center));
    cv::threshold(roi, roi, thres, 255, cv::THRESH_BINARY);
    for (int i = ver_center; i < depths_ocv.rows; i += step) {
        cv::Mat roi = depths_ocv(cv::Rect(0, i, depths_ocv.cols, step));
        cv::threshold(roi, roi, thres, 255, cv::THRESH_BINARY);
        thres += 3;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //cv::imshow("Depth", depths_ocv);
//green_channel = green_channel - depths_ocv;
    cv::Mat floor_depth = green_channel.clone();
    // Normal maps threshold (Heuristic threshold values)
    int floor_thres = 100;
    int obstacle_thres = 150;
    int wall_thres = 180;
    cv::threshold(green_channel, green_channel, floor_thres, 255, cv::THRESH_BINARY);
    cv::threshold(red_channel, red_channel, obstacle_thres, 255, cv::THRESH_BINARY);
    cv::threshold(blue_channel, blue_channel, wall_thres, 255, cv::THRESH_BINARY);

//depths_ocv = depths_ocv - blue_channel;

    // Remove values beyond depth detections using logical operations
    cv::bitwise_and(red_channel, depths_ocv, red_channel);
    cv::bitwise_and(floor_depth, depths_ocv, floor_depth);
    cv::bitwise_and(green_channel, depths_ocv, green_channel);
    cv::bitwise_and(blue_channel, depths_ocv, blue_channel);

    //cv::imshow("X Thresh", blue_channel);
    //cv::imshow("Y Thresh", green_channel);
    //cv::imshow("Z Thresh", red_channel);
    //cv::imshow("Image Orig", pixels);

    /////////////////// Count how many pixels are in each grid position /////////////////////
    
    const float side_percent = 0.35;
    const float top_percent = 0;
    
    // Calculate margins
    int side_margin = int(image_size.width * side_percent);
    int top_margin = int(image_size.height * top_percent);

    // Size of each block in pixels
    int block_wide = int((1 - side_percent * 2) * image_size.width / NUM_BLOCKS);
    int block_height = image_size.height / NUM_ROWS;
	
    //Calculate min depth
    float depth_value;
    float min_depth[10];
    
    for (int j = 0; j < NUM_ROWS; j++) {
        for (int k = 0; k < NUM_BLOCKS; k++) {
           object_pixels[j][k] = 0;
	   dist_object[j][k] = 0;
            // Loop over the image in row major for stairs and hole detection
            for (int col = side_margin + k * block_wide; col < side_margin + (k + 1) * block_wide; col++) {
                int end_row = block_height * (j + 1);
                int begin_row = end_row - block_height;
                for (int row = end_row - 1; row > begin_row; row--) {
		    // Paint wall pixels
                    if (blue_channel.data[(blue_channel.cols * row) + col] == 255) {
			pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels())] = 100;
			pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels()) + 1] = 0;
			pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels()) + 2] = 0;
                    }
		    // Paint floor pixels
                    if (green_channel.data[(green_channel.cols * row) + col] == 255) {
			pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels())] = 0;
			pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels()) + 1] = 100;
			pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels()) + 2] = 0;
                    }
		    // Paint obstacle pixels
                    if (red_channel.data[(red_channel.cols * row) + col] == 255) {
                        pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels())] = 0;
                        pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels()) + 1] = 0;
                        pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels()) + 2] = 150;
			float aux;
			if(depth_value>min_depth[9]){
				min_depth[9] = depth_value;
				for (int i = 0;i < 10; i++){
					for (int j = 0; j< 9; j++){
						if (min_depth[j] < min_depth[j+1]){ 
							aux = min_depth[j]; 
							min_depth[j] = min_depth[j+1]; 
							min_depth[j+1] = aux;
						}
					}
				}
			}
                         object_pixels[j][k]++;
                    }                
                } // for rows
            } // for cols
	    for(int m = 0; m<10;m++){
		depth_value = depth_value + min_depth[m];
	    }
	    depth_value = depth_value/10;
            dist_object[j][k] = depth_value;
        } // for NUM_BLOCKS 
    } // for NUM_ROWS
    
    int temp_x_step = stepDetection(side_margin, block_wide, floor_depth);
    
    // Safe x_step assignation
    assert(pthread_mutex_lock(&step_mutex) == 0);
    x_step = temp_x_step;
    assert(pthread_mutex_unlock(&step_mutex) == 0);
 
    //cv::imshow("Depth", orig_depths);
    //cv::imshow("Floor", green_channel);
    //cv::bitwise_and(orig_depths, orig_depths, green_channel);
    //cv::imshow("Masked", orig_depths);
    
    // Paint grid
    cv::line(pixels, cv::Point(side_margin, 0), cv::Point(side_margin, pixels.rows), cv::Scalar(0, 0, 255), 2);
    cv::line(pixels, cv::Point(pixels.cols - side_margin, 0), cv::Point(pixels.cols - side_margin, pixels.rows), cv::Scalar(0, 0, 255), 2);
    cv::line(pixels, cv::Point(side_margin + block_wide, 0), cv::Point(side_margin + block_wide, pixels.rows), cv::Scalar(0, 0, 255), 2);
    cv::line(pixels, cv::Point(side_margin + 2 * block_wide, 0), cv::Point(side_margin + 2 * block_wide, pixels.rows), cv::Scalar(0, 0, 255), 2);
    cv::line(pixels, cv::Point(side_margin, block_height - top_margin), cv::Point(pixels.cols - side_margin, block_height - top_margin), cv::Scalar(0, 0, 255), 2);
}

/*
void* _process(void*) {
    // Calculate margins
    int side_margin = int(image_size.width * side_percent);
    int top_margin = int(image_size.height * top_percent);

    // Size of each block in pixels
    int block_wide = int((1 - side_percent * 2) * image_size.width / NUM_BLOCKS);

    // Size in pixels of the previously detected object
    int previous_c = 0;
    // Array position of the previously detected object
    int previous_row = 0;
    int previous_col = 0;

    int block_height = image_size.height / NUM_ROWS;
    std::vector<std::vector<int> > object_pixels(NUM_ROWS, std::vector<int> (NUM_BLOCKS));
    float prev_depth_value = 0;
    int xend;

    // local version of shared variables
    float max_dist;
    float min_dist;
    int threshold;
    cv::Mat pixels;

    sl::Mat d;
    sl::Mat n;

    while (get_key() != 'q') {
        /////////// Step 2c: Image Processing ///////////////
        int c_max = 0;
        int row_max = -1;
        int col_max = -1;
        /////////// Step 2c.1: Wait for new frame ///////////////
        // if (get_key() == 'q') break;
        mysem_wait(&sem);
        assert(pthread_mutex_lock(&zed_mutex) == 0);
        zed.retrieveMeasure(depths, sl::MEASURE_DEPTH);
        depths.copyTo(d);
        zed.retrieveMeasure(normals, sl::MEASURE_NORMALS);
        normals.copyTo(n);
        assert(pthread_mutex_unlock(&zed_mutex) == 0);

        assert(pthread_mutex_lock(&min_dist_mutex) == 0);
        min_dist = min_distance;
        assert(pthread_mutex_unlock(&min_dist_mutex) == 0);

        assert(pthread_mutex_lock(&max_dist_mutex) == 0);
        max_dist = max_distance;
        assert(pthread_mutex_unlock(&max_dist_mutex) == 0);

        assert(pthread_mutex_lock(&threshold_mutex) == 0);
        threshold = detection_threshold;
        assert(pthread_mutex_unlock(&threshold_mutex) == 0);

        pixels = cv::Mat::zeros(image_size.height, image_size.width, 24);

        /////////// Step 2c.2: Calculate normals ///////////////
        assert(pthread_mutex_lock(&calibrate_mutex) == 0);
        int cal = calibrate_flag;
        calibrate_flag = false;
        assert(pthread_mutex_unlock(&calibrate_mutex) == 0);
        if (cal) {
            floor_normals = calibrate(n);
        }
        for (int j = 0; j < NUM_ROWS; j++) {
            for (int k = 0; k < NUM_BLOCKS; k++) {
                object_pixels[j][k] = 0;
                // Loop over the image in row major for stairs and hole detection
                for (int col = side_margin + k * block_wide; col < side_margin + (k + 1) * block_wide; col++) {
                    int end_row = block_height * (j + 1);
                    int begin_row = end_row - block_height;
                    for (int row = end_row - 1; row > begin_row; row--) {
                        //////////// Step 2c.3: Depth information /////////////////
                        float depth_value = 0;
                        d.getValue(col, row, &depth_value);
                        // If the difference between the current depth value and the previous value is greater than 5 cms
                        // There is a step
                        if ((depth_value - prev_depth_value) > 5) {
                            pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels()) + 2] = 255;
                        }
                        prev_depth_value = depth_value;

                        /////////// Step 2c.4: Normal information ////////////////
                        sl::float4 pixel_normals;
                        n.getValue(col, row, &pixel_normals);
                        /////////// Step 2c.5: Obstacle detection ////////////////
                        // Limit maximum detection distance in top row
                        int max_distance = max_dist;
                        if (j == 0) max_distance = 150;
                        // depth_value == depth_value to avoid NaN values
                        if (depth_value == depth_value and depth_value < max_distance and depth_value > min_dist and
                                std::abs(std::abs(floor_normals[0]) - std::abs(pixel_normals[0])) > 0.2 and
                                std::abs(std::abs(floor_normals[1]) - std::abs(pixel_normals[1])) > 0.2 and
                                std::abs(std::abs(floor_normals[2]) - std::abs(pixel_normals[2])) > 0.2) {

                            // Color of the obstacle based on distance to observer
                            if (depth_value > min_dist and depth_value < max_dist * 0.3) {
                                pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels()) + 2] = 255;
                            } else if (depth_value >= max_dist * 0.3 and depth_value < max_dist * 0.6) {
                                pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels()) + 2] = 255;
                                pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels()) + 1] = 255;
                            } else if (depth_value >= max_dist * 0.6) {
                                pixels.data[(pixels.cols * row * pixels.channels()) + (col * pixels.channels()) + 1] = 255;
                            }
                            object_pixels[j][k]++;
                        }
                    } // for rows
                } // for cols
                // Increase threshold in lateral columns
                if (k == 0 || k == NUM_BLOCKS - 1) threshold = threshold * 1.2;
                // Save biggest object information
                if (object_pixels[j][k] >= threshold and object_pixels[j][k] > c_max) {
                    c_max = object_pixels[j][k];
                    row_max = j;
                    col_max = k;
                }
            } // for NUM_BLOCKS 
        } // for NUM_ROWS
        /////////// Step 2c.6: Visualization update ////////////////
        // Paint grid
        cv::line(pixels, cv::Point(side_margin, 0), cv::Point(side_margin, pixels.rows), cv::Scalar(0, 0, 255), 2);
        cv::line(pixels, cv::Point(pixels.cols - side_margin, 0), cv::Point(pixels.cols - side_margin, pixels.rows), cv::Scalar(0, 0, 255), 2);
        cv::line(pixels, cv::Point(side_margin + block_wide, 0), cv::Point(side_margin + block_wide, pixels.rows), cv::Scalar(0, 0, 255), 2);
        cv::line(pixels, cv::Point(side_margin + 2 * block_wide, 0), cv::Point(side_margin + 2 * block_wide, pixels.rows), cv::Scalar(0, 0, 255), 2);
        cv::line(pixels, cv::Point(side_margin, block_height - top_margin), cv::Point(pixels.cols - side_margin, block_height - top_margin), cv::Scalar(0, 0, 255), 2);
        // copy detection to shared pixels
        pixeles[pixelesgrab] = pixels;
        updatePixelesGrab();

        // Make the arrow point to the object
        // avoid vibration of the arrow
        if (abs(c_max - previous_c) < threshold) {
            row_max = previous_row;
            col_max = previous_col;
        }

        // safe assignement of x_end
        assert(pthread_mutex_lock(&object_mutex) == 0);
        object_row = row_max;
        object_col = col_max;
        assert(pthread_mutex_unlock(&object_mutex) == 0);

        previous_c = c_max;
        previous_row = row_max;
        previous_col = col_max;
        /////////// END Step 2b: Image Processing ///////////////
    }
    pthread_exit(NULL);
}

void* processUD(void*) {
    unsigned int MAX_DISP = 255;
    float value_ranges[] = {(float) 0, (float) MAX_DISP};
    const float* hist_ranges[] = {value_ranges};
    int channels[] = {0};
    int histSize[] = {MAX_DISP};
    sl::Mat d;
    while (get_key() != 'q') {
        if (get_key() == 'q') break;
        mysem_wait(&sem);
        assert(pthread_mutex_lock(&zed_mutex) == 0);
        zed.retrieveMeasure(depths, sl::MEASURE_DEPTH);
        depths.copyTo(d);
        assert(pthread_mutex_unlock(&zed_mutex) == 0);
        cv::Mat img = slMat2cvMat(d);

        normalizeMat(img);
        img.convertTo(img, CV_8U, 255);
        //cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
        cv::imshow("depth", img);

        cv::Mat uhist = cv::Mat::zeros(MAX_DISP, img.cols, CV_32F);
        cv::Mat vhist = cv::Mat::zeros(img.rows, MAX_DISP, CV_32F);
        cv::Mat tmpImageMat, tmpHistMat;
        // === CALCULATE V-HIST ===
        for (int i = 0; i < img.rows; i++) {
            tmpImageMat = img.row(i);
            vhist.row(i).copyTo(tmpHistMat);
            cv::calcHist(&tmpImageMat, 1, channels, cv::Mat(), tmpHistMat, 1, histSize, hist_ranges, true, false);
            vhist.row(i) = tmpHistMat.t() / (float) img.rows;
        }

        // === CALCULATE U-HIST ===
        for (int i = 0; i < img.cols; i++) {
            tmpImageMat = img.col(i);
            uhist.col(i).copyTo(tmpHistMat);
            cv::calcHist(&tmpImageMat, 1, channels, cv::Mat(), tmpHistMat, 1, histSize, hist_ranges, true, false);
            uhist.col(i) = tmpHistMat / (float) img.cols;
        }
        uhist.convertTo(uhist, CV_8U, 255);
        //cv::flip(uhist, uhist, 0);
        uhist = uhist * 10;
        vhist.convertTo(vhist, CV_8U, 255);
        //cv::flip(vhist, vhist, 1);
        vhist = vhist * 10;
        //cv::Sobel(uhist, uhist, CV_8U, 0, 1);
        //cv::applyColorMap(uhist, uhist, cv::COLORMAP_JET);

        cv::imshow("uhist", uhist);
        cv::imshow("vhist", vhist);
        cv::waitKey(1);
    } // while not 'q'
}
*/
sl::float4 calibrate(sl::Mat& normals) {
    // Tracking currently not used
    // zed.enableTracking(tracking_parameters);

    // Set x coordinate to the frame center
    int x = normals.getWidth() / 2;

    // Set y coordinates to the bottom part of the image
    int y_ini = (normals.getHeight() / 4) * 3;
    int y_end = normals.getHeight();

    sl::float4 total = sl::float4(sl::Vector4<float>(0, 0, 0, 0));
    int samples = 0;

    ///////// Measure the average floor normals /////////
    // Sum all normal values along y axis
    for (int i = y_ini; i < y_end; i++) {
        sl::float4 val;
        normals.getValue(i, x, &val);
        if (!std::isnan(val[0])) {
            total += val;
            samples++;
        }
    }
    // Average those values
    total /= samples;
    cont_calib++;
    return total;
}

// Remove NaN values and normalize between 0.0 and 1.0  
void normalizeMat(cv::Mat& input) {
    float min_dist = 30;
    float max_dist = 768;
    for(int row = 0; row < input.rows; row++) {
        for(int col = 0; col < input.cols; col++) {
            if (input.at<float>(row, col) != input.at<float>(row, col)) //NaN doesn't equal NaN
                input.at<float>(row, col) = 0;
            else if (input.at<float>(row, col) > max_distance)
                input.at<float>(row, col) = 0;
            else if (input.at<float>(row, col) < min_distance)
                input.at<float>(row, col) = 1;
            else
                input.at<float>(row, col) = 1 - ((input.at<float>(row, col) - min_dist) / max_dist);
        }
    }
}

int stepDetection(int side_margin, int block_wide, cv::Mat& orig_depths) {
    cv::Mat depth_roi = orig_depths(cv::Rect(side_margin + block_wide, 0, block_wide, orig_depths.rows));
    cv::Mat tmpImageMat, tmpHistMat;

    int MAX_DISP = 255;
    float value_ranges[] = {(float)0, (float)MAX_DISP};
    const float* hist_ranges[] = {value_ranges};
    int channels[] = {0};
    int histSize[] = {MAX_DISP};

    cv::Mat vhist = cv::Mat::zeros(depth_roi.rows, MAX_DISP, CV_32F); 
    // === CALCULATE V-HIST ===
         for(int i = (depth_roi.rows / 2) - 1; i < depth_roi.rows; i++)
          {
            tmpImageMat = depth_roi.row(i);
            vhist.row(i).copyTo(tmpHistMat);
            cv::calcHist(&tmpImageMat, 1, channels, cv::Mat(), tmpHistMat, 1, histSize, hist_ranges, true, false);
            vhist.row(i) = tmpHistMat.t() / (float) depth_roi.rows / 2;
          }
    vhist = vhist * 20;
    vhist.convertTo(vhist, CV_8U, 255);
    cv::cvtColor(vhist, vhist, cv::COLOR_GRAY2BGR);
    cv::imshow("vhist - original", vhist);	
    
    //int value = MaxDiff(vhist);           //Madrid version
    int value = lineRegression(vhist);  //Sutter version
    
    //cv::imshow("vhist - evaluation", vhist);	//shows line segmentation
    
    return value;
}

int lineRegression(cv::Mat& img){
    //Written by Sutter Lum
    //should return -1 if nothing
    //returns y-position of x_step otherwise
    
    int MCLocation = -1;   //y-location of largest disparity
    int NCLocation = -1;   //y-location of current disparity
    int maxChange = -1; //biggest disparity from line
    int newChange = -1; //current disparity from line
    
    std::vector<cv::Mat> contours;
    cv::Mat grey; 
    cv::Mat thresh;
    cv::Mat resultLine;
    
    cv::cvtColor(img, grey,cv::COLOR_BGR2GRAY);
    cv::threshold(grey, thresh, 19, 255, cv::THRESH_BINARY); //turns image into binary
    cv::findContours(thresh, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    //Drawing contours --------------------------------------------------------------------
    cv::Mat drawing = cv::Mat::zeros( thresh.size(), CV_8UC3 );
    cv::drawContours(drawing, contours, -1,255, 1, 8);

    //cv::imshow( "Contours", drawing );
    //---------------------------------------------------------------------------------------

    //Sutter
    for(int t = 0; t < contours.size(); t++){
        cv::fitLine(contours[t], resultLine, cv::DIST_L2, 0, 0.01, 0.01);
	//std::cout << "Size Contour: " << contours[t].size()<< "Contour Value: "<< contours[t] <<"\n";
    }
    //bool verif = (contours[1].size() < contours[10].size());
    //std::cout << " Verdad? " << contours[1].size() <<"\n";
    //std::cout << " Cols: " << contours[1].cols <<"\n";
    //std::cout << " Rows? " << contours[1].rows <<"\n";    
//End -- Sutter

    /* El que volem es que de tots el contours trobats, et busqui el mes gran i d'alla fagi la regressió
    int controur_max_index = 0;
    for(int t = 0; t < contours.size(); t++){
        if (contours[t].size()>contours[controur_max_index].size()){
	    controur_max_index = t;
        }
    }
    cv::fitLine(contours[controur_max_index], resultLine, cv::DIST_L2, 0, 0.01, 0.01);
    */

    cv::Point pt2, pt1;
    float pendent, n;
    pendent =  resultLine.at<float>(0,1)/resultLine.at<float>(0,0);
    n = resultLine.at<float>(0,3)-pendent*resultLine.at<float>(0,2);
    pt2.x = 0;
    pt2.y = pendent*resultLine.at<float>(0,2) + n;
    pt1.x = img.rows-1;
    pt1.y = pendent*pt1.x+n;

    /*proba de punt
    pt1.x = 0;
    pt1.y = img.rows-1;
    pt2.x = img.cols/2-1;
    pt2.y = img.rows/2;
    */
/*
    std::cout << "Image cols: " << img.cols << "\n";
    std::cout << "\n";
    std::cout << "Size: " << resultLine.size() << "\n";
    std::cout << "Rows: " << resultLine.rows << "\n";
    std::cout << "Cols: " << resultLine.cols << "\n";
    std::cout << "ResultLine 1: " << resultLine.at<float>(0,0)<< "\n";
    std::cout << "ResultLine 2: " << resultLine.at<float>(1,0)<< "\n";
    std::cout << "ResultLine 3: " << resultLine.at<float>(2,0)<< "\n";
    std::cout << "ResultLine 4: " << resultLine.at<float>(3,0)<< "\n";
*/
    //cv::circle(drawing, pt1, 2, 255, 2);
    std::cout << "Aixo que es???? " << resultLine.at<float>(3, 0, 0);
    std::cout << "Aixo que es???? " << tuyperesultLine.at<float>(3, 0, 0);
    cv::line(drawing, pt1 ,pt2, 255, 2);
    cv::imshow( "Contours", drawing );


    //ONLY loops through image IF the fitLine has a slope to the right (negative disparity)
    int lineSlope = checkSlope(resultLine);

    if (lineSlope == 1){
    
        //Loops through histogram and average line images
        //compares the difference between the individual points and the average line
        //stores the point farthest from the average line (if it's not on the edge)
        int firstPOI = -1;
        int secondPOI = -1;
    
        for (int i = (img.rows-1); i > (img.rows / 2); i = i - 5){
            if(secondPOI != -1 && ((firstPOI - secondPOI) > 150) && ((firstPOI - secondPOI) < 230)) { //only records a change if there are two points to compare within a reasonable distance
                newChange = (firstPOI - secondPOI);
            }
            NCLocation = i; //updates y-location each row
            firstPOI = -1; //comparison points reset every row
            secondPOI = -1;
        
            for (int j = 0; j < img.cols; j++) { //goes across the images
                if(img.at<uchar>(i, j, 0) != resultLine.at<uchar>(i, j, 0)){
                    if(firstPOI != -1){
                        secondPOI = img.at<uchar>(i, j, 0);
                    }
                    else{
                        firstPOI = img.at<uchar>(i, j, 0);
                    }
                }
                if (newChange > maxChange){ //only stores the largest disparity
                    maxChange = newChange;
                    newChange = -1;
                    MCLocation = NCLocation;
                    /****************************************************
                    //marks the second v-hist with the segmentation line
                    for (int i = 0; i < img.rows; i++){
                        img.at<uchar>(i, MCLocation, 0) = 255; 
                    }
                    ****************************************************/
                }
            }
        }
        x_step = MCLocation;
    }
    //printf("\nx_step value: %d", x_step);
    //printf("\nmaxChange value: %d\n", maxChange);
    return x_step;
}

int MaxDiff(cv::Mat& img) {
    int prev_pos = -1;
    int diff_total = 0;
    int max_diff = -1;
    int max_pos = -1;
    for(int i = (img.rows-1); i > (img.rows / 2); i = i - 5) {  //Loops through each row of the histogram in reverse order, stopping halfway
        int max = -1;
        int pos = -1;
        for (int j = 0; j < img.cols; j++) {
            uchar current = img.at<uchar>(i, j, 0);
            if (current >= max) {
                max = current;
                pos = j;
            }
        }
        
        if (prev_pos != -1) {
            diff_total += abs(prev_pos - pos);
        }
        if (max_diff < abs(prev_pos - pos)) {
            max_diff = abs(prev_pos - pos);
            max_pos = i;
        }
        prev_pos = pos;
    }
    //float average = diff_total / float(img.rows);
    //if (max_diff > (average * 10)) {
    if (max_diff > 45  && max_diff<225) {
        x_step = max_pos;
    }
    return x_step;
}

int checkSlope(cv::Mat& img){
    //loops through the resultLine Matrix and stores the first and last point
    int result;
    int firstPointX,firstPointY;
    int lastPointX, lastPointY;
   
   for (int i = (img.rows - 1); i > (img.rows / 2); i = i - 5){
        for (int j = 0; j < img.cols; j++) {
            if(img.at<uchar>(i, j, 0) != 0){
                if(firstPointX == 0 && firstPointY == 0){
                    //only takes the firstPoint data once
                    firstPointX = i;
                    firstPointY = j;
                }
                else{
                    //constantly updates lastPoint data until it scans the whole line
                    lastPointX = i;
                    lastPointY = j;
                }
            }
        }
   }
    int slope = ((lastPointY - firstPointY) / (lastPointX - firstPointX));

    if (slope > 0){ //positive slope means negative disparity
        result = 1;
    }
    //printf("\nslope: %d", slope);
    return result;
}

