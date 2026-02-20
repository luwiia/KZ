#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// ================== СВОИ РЕАЛИЗАЦИИ ==================

// Своя реализация BGR -> Grayscale
Mat myBGR2GRAY(const Mat& bgr) {
    Mat gray(bgr.size(), CV_8UC1);
    
    for (int i = 0; i < bgr.rows; i++) {
        for (int j = 0; j < bgr.cols; j++) {
            Vec3b pixel = bgr.at<Vec3b>(i, j);
            // Формула: 0.299*R + 0.587*G + 0.114*B
            uchar b = pixel[0];
            uchar g = pixel[1];
            uchar r = pixel[2];
            
            uchar gray_val = static_cast<uchar>(
                0.114 * b + 0.587 * g + 0.299 * r
            );
            gray.at<uchar>(i, j) = gray_val;
        }
    }
    return gray;
}

// Своя реализация BGR -> HSV
Mat myBGR2HSV(const Mat& bgr) {
    Mat hsv(bgr.size(), CV_8UC3);
    
    for (int i = 0; i < bgr.rows; i++) {
        for (int j = 0; j < bgr.cols; j++) {
            Vec3b pixel = bgr.at<Vec3b>(i, j);
            
            // Нормализация к [0,1]
            float b = pixel[0] / 255.0;
            float g = pixel[1] / 255.0;
            float r = pixel[2] / 255.0;
            
            float max_val = max(max(r, g), b);
            float min_val = min(min(r, g), b);
            float delta = max_val - min_val;
            
            // Hue (оттенок)
            float h = 0;
            if (delta != 0) {
                if (max_val == r) {
                    h = 60 * fmod((g - b) / delta, 6);
                } else if (max_val == g) {
                    h = 60 * ((b - r) / delta + 2);
                } else { // max_val == b
                    h = 60 * ((r - g) / delta + 4);
                }
            }
            if (h < 0) h += 360;
            
            // Saturation (насыщенность)
            float s = (max_val == 0) ? 0 : (delta / max_val);
            
            // Value (значение)
            float v = max_val;
            
            // Конвертация в 8-битные значения OpenCV
            uchar h_8bit = static_cast<uchar>(h / 2); // OpenCV: 0-180
            uchar s_8bit = static_cast<uchar>(s * 255);
            uchar v_8bit = static_cast<uchar>(v * 255);
            
            hsv.at<Vec3b>(i, j) = Vec3b(h_8bit, s_8bit, v_8bit);
        }
    }
    return hsv;
}

// Функция для попиксельного сравнения
void compareImages(const Mat& img1, const Mat& img2, const string& name) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        cout << "Изображения разного размера или типа!" << endl;
        return;
    }
    
    double diff = 0;
    int total_pixels = img1.rows * img1.cols * img1.channels();
    
    for (int i = 0; i < img1.rows; i++) {
        for (int j = 0; j < img1.cols; j++) {
            if (img1.type() == CV_8UC1) {
                // Grayscale
                diff += abs(img1.at<uchar>(i, j) - img2.at<uchar>(i, j));
            } else {
                // Цветное
                Vec3b p1 = img1.at<Vec3b>(i, j);
                Vec3b p2 = img2.at<Vec3b>(i, j);
                diff += abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2]);
            }
        }
    }
    
    double avg_diff = diff / total_pixels;
    cout << "Средняя разница для " << name << ": " << avg_diff << endl;
    if (avg_diff < 1.0) {
        cout << "✓ Реализация корректна!" << endl;
    } else {
        cout << "✗ Есть расхождения" << endl;
    }
}

// ================== ОСНОВНЫЕ ФУНКЦИИ ==================

// Пункт 1: Загрузка и сохранение изображения
void task1() {
    cout << "\n=== ЗАДАНИЕ 1: Загрузка и сохранение изображения ===" << endl;
    
    // Загрузка изображения
    Mat img = imread("test_image.jpg");
    if (img.empty()) {
        cout << "Файл test_image.jpg не найден. Создаю тестовое изображение..." << endl;
        // Создаем тестовое изображение
        img = Mat(400, 400, CV_8UC3, Scalar(255, 0, 0)); // Синий квадрат
        // Добавим несколько цветных объектов
        rectangle(img, Rect(100, 100, 100, 100), Scalar(0, 255, 0), FILLED); // Зеленый квадрат
        rectangle(img, Rect(250, 250, 100, 100), Scalar(0, 0, 255), FILLED); // Красный квадрат
        circle(img, Point(200, 300), 50, Scalar(255, 255, 0), FILLED); // Желтый круг
        
        imwrite("test_image.jpg", img);
        cout << "Создан файл test_image.jpg" << endl;
    }
    
    // Отображение
    imshow("Задание 1: Исходное изображение", img);
    cout << "Размер изображения: " << img.cols << "x" << img.rows << endl;
    cout << "Нажмите любую клавишу для продолжения..." << endl;
    waitKey(0);
    
    // Сохранение в PNG
    imwrite("test_image_output.png", img);
    cout << "Изображение сохранено как test_image_output.png" << endl;
}

// Пункт 2: Захват видео с камеры и из файла
void task2() {
    cout << "\n=== ЗАДАНИЕ 2: Захват видео ===" << endl;
    
    // Часть 1: Веб-камера
    cout << "Запуск веб-камеры (ID=0). Нажмите 'q' для выхода..." << endl;
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Не удалось открыть камеру!" << endl;
    } else {
        Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            imshow("Камера (нажмите 'q' для выхода)", frame);
            
            if (waitKey(30) == 'q') break;
        }
        cap.release();
    }
    
    // Часть 2: Видеофайл
    cout << "\nЗапуск видео из файла. Нажмите 'q' для выхода..." << endl;
    VideoCapture video("test_video.mp4");
    if (!video.isOpened()) {
        cout << "Файл test_video.mp4 не найден. Создаю тестовое видео..." << endl;
        // Создадим тестовое видео из изображений
        Mat img = imread("test_image.jpg");
        if (!img.empty()) {
            VideoWriter writer("test_video.mp4", 
                              VideoWriter::fourcc('m', 'p', '4', 'v'), 
                              30, img.size());
            for (int i = 0; i < 30; i++) { // 1 секунда видео
                writer.write(img);
            }
            writer.release();
            cout << "Создан файл test_video.mp4" << endl;
            video.open("test_video.mp4");
        }
    }
    
    if (video.isOpened()) {
        Mat frame;
        while (true) {
            video >> frame;
            if (frame.empty()) break;
            
            imshow("Видеофайл (нажмите 'q' для выхода)", frame);
            
            if (waitKey(30) == 'q') break;
        }
        video.release();
    }
    
    destroyAllWindows();
}

// Пункт 3: Конвертация цветовых пространств
void task3() {
    cout << "\n=== ЗАДАНИЕ 3: Конвертация цветовых пространств ===" << endl;
    
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Не удалось открыть камеру!" << endl;
        return;
    }
    
    cout << "Запуск камеры. Нажмите 'q' для выхода..." << endl;
    cout << "Окна: Оригинал | Grayscale (OpenCV) | Grayscale (моя) | HSV (OpenCV) | HSV (моя)" << endl;
    
    Mat frame, gray_opencv, hsv_opencv;
    Mat my_gray, my_hsv;
    
    int frame_count = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // OpenCV методы
        cvtColor(frame, gray_opencv, COLOR_BGR2GRAY);
        cvtColor(frame, hsv_opencv, COLOR_BGR2HSV);
        
        // Свои реализации
        my_gray = myBGR2GRAY(frame);
        my_hsv = myBGR2HSV(frame);
        
        // Показываем все окна
        imshow("1. Оригинал (BGR)", frame);
        imshow("2. Grayscale (OpenCV)", gray_opencv);
        imshow("3. Grayscale (моя реализация)", my_gray);
        imshow("4. HSV (OpenCV)", hsv_opencv);
        imshow("5. HSV (моя реализация)", my_hsv);
        
        // Сравнение каждые 30 кадров
        if (frame_count % 30 == 0) {
            cout << "\n--- Сравнение результатов (кадр " << frame_count << ") ---" << endl;
            compareImages(gray_opencv, my_gray, "Grayscale");
            compareImages(hsv_opencv, my_hsv, "HSV");
        }
        
        frame_count++;
        
        if (waitKey(30) == 'q') break;
    }
    
    cap.release();
    destroyAllWindows();
}

// Пункт 4: Сегментация по цвету с inRange и bitwise_and
void task4() {
    cout << "\n=== ЗАДАНИЕ 4: Сегментация по цвету ===" << endl;
    
    Mat img = imread("test_image.jpg");
    if (img.empty()) {
        cout << "Ошибка загрузки изображения!" << endl;
        return;
    }
    
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // Диапазон для синего цвета
    Scalar lower_blue(100, 50, 50);
    Scalar upper_blue(140, 255, 255);
    
    Mat mask;
    inRange(hsv, lower_blue, upper_blue, mask);
    
    Mat result;
    bitwise_and(img, img, result, mask);
    
    // Подсчет количества пикселей синего цвета
    int blue_pixels = countNonZero(mask);
    cout << "Найдено пикселей синего цвета: " << blue_pixels << endl;
    cout << "Процент синего цвета: " << (100.0 * blue_pixels / (img.rows * img.cols)) << "%" << endl;
    
    imshow("Задание 4: Исходное изображение", img);
    imshow("Задание 4: Маска (синий)", mask);
    imshow("Задание 4: Результат (только синий)", result);
    
    cout << "Нажмите любую клавишу для продолжения..." << endl;
    waitKey(0);
    destroyAllWindows();
}

// Пункт 5: Гистограмма для канала Hue
void task5() {
    cout << "\n=== ЗАДАНИЕ 5: Гистограмма Hue канала ===" << endl;
    
    Mat img = imread("test_image.jpg");
    if (img.empty()) return;
    
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // Разделяем каналы
    vector<Mat> channels;
    split(hsv, channels);
    Mat hue = channels[0]; // Канал H
    
    // Параметры гистограммы
    int histSize = 180; // Hue от 0 до 179
    float range[] = {0, 180};
    const float* histRange = {range};
    
    Mat hist;
    calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    
    // Визуализация гистограммы
    int hist_w = 512, hist_h = 400;
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
    
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX);
    
    // Рисуем гистограмму линиями
    for (int i = 1; i < histSize; i++) {
        line(histImage,
             Point((i-1) * hist_w / histSize, 
                   hist_h - cvRound(hist.at<float>(i-1))),
             Point(i * hist_w / histSize, 
                   hist_h - cvRound(hist.at<float>(i))),
             Scalar(255, 0, 0), 2);
    }
    
    // Добавим сетку для удобства
    for (int i = 0; i <= 180; i += 30) {
        int x = i * hist_w / 180;
        line(histImage, Point(x, 0), Point(x, hist_h), Scalar(200, 200, 200), 1);
    }
    
    // Найдем пики гистограммы
    cout << "Анализ гистограммы:" << endl;
    double max_val;
    Point max_loc;
    minMaxLoc(hist, 0, &max_val, 0, &max_loc);
    
    cout << "Максимальный пик: H=" << max_loc.y << " (значение=" << max_val << ")" << endl;
    
    // Определим доминирующие цвета по пикам
    cout << "Доминирующие оттенки:" << endl;
    for (int h = 0; h < 180; h += 10) {
        float val = hist.at<float>(h);
        if (val > max_val * 0.5) { // Больше 50% от максимума
            cout << "  H≈" << h << "-" << (h+10) << ": значительный пик" << endl;
        }
    }
    
    imshow("Задание 5: Исходное изображение", img);
    imshow("Задание 5: Гистограмма Hue", histImage);
    
    cout << "Нажмите любую клавишу для продолжения..." << endl;
    waitKey(0);
    destroyAllWindows();
}

// Пункт 6: Сегментация для 3 цветов
void task6() {
    cout << "\n=== ЗАДАНИЕ 6: Сегментация для 3 цветов ===" << endl;
    
    Mat img = imread("test_image.jpg");
    if (img.empty()) return;
    
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // Оптимальные диапазоны для разных цветов (подобраны экспериментально)
    struct ColorRange {
        string name;
        Scalar lower;
        Scalar upper;
        Scalar display_color;
    };
    
    vector<ColorRange> colors = {
        {"Синий",  Scalar(100, 50, 50),  Scalar(140, 255, 255), Scalar(255, 0, 0)},
        {"Зеленый", Scalar(40, 50, 50),  Scalar(80, 255, 255),  Scalar(0, 255, 0)},
        {"Красный", Scalar(0, 50, 50),   Scalar(10, 255, 255),  Scalar(0, 0, 255)},
        // Для красного нужен второй диапазон
        {"Красный2", Scalar(160, 50, 50), Scalar(180, 255, 255), Scalar(0, 0, 255)}
    };
    
    Mat combined_result = Mat::zeros(img.size(), img.type());
    vector<Mat> results;
    
    for (const auto& color : colors) {
        Mat mask, result;
        inRange(hsv, color.lower, color.upper, mask);
        bitwise_and(img, img, result, mask);
        
        results.push_back(result);
        
        // Добавляем в общий результат
        combined_result += result;
        
        // Статистика
        int pixels = countNonZero(mask);
        cout << color.name << ": " << pixels << " пикселей ("
             << (100.0 * pixels / (img.rows * img.cols)) << "%)" << endl;
        
        // Показываем отдельно для каждого цвета
        imshow("Сегментация: " + color.name, result);
    }
    
    imshow("Задание 6: Исходное изображение", img);
    imshow("Задание 6: Все цвета вместе", combined_result);
    
    cout << "\nОптимальные диапазоны HSV:" << endl;
    cout << "Синий:    H:100-140, S:50-255, V:50-255" << endl;
    cout << "Зеленый:  H:40-80,   S:50-255, V:50-255" << endl;
    cout << "Красный:  H:0-10 и 160-180, S:50-255, V:50-255" << endl;
    
    cout << "\nНажмите любую клавишу для завершения..." << endl;
    waitKey(0);
    destroyAllWindows();
}

// ================== ГЛАВНАЯ ФУНКЦИЯ ==================

int main() {
    cout << "==========================================" << endl;
    cout << "   ЛАБОРАТОРНАЯ РАБОТА 1: OpenCV" << endl;
    cout << "==========================================" << endl;
    
    // Выполняем все задания по порядку
    task1();  // Загрузка и сохранение изображения
    task2();  // Захват видео с камеры и из файла
    task3();  // Конвертация цветовых пространств
    task4();  // Сегментация по цвету
    task5();  // Гистограмма Hue
    task6();  // Сегментация для 3 цветов
    
    cout << "\n==========================================" << endl;
    cout << "   ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА" << endl;
    cout << "==========================================" << endl;
    
    return 0;
}
