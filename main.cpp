#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include <opencv2\opencv.hpp>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <stack>
#include <queue>

using namespace cv;
using namespace std;



Mat img, inpaintMask;
const float sigmac = 50.0, lambda = 0.09, lambdat = 0.99;
const int dxy[4][2] = { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } };
const int dxy8[8][2] = { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 }, { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 } };
/***********************************inpainting*********************************************************/

#undef CV_MAT_ELEM_PTR_FAST
#define CV_MAT_ELEM_PTR_FAST( mat, row, col, pix_size )  \
	((mat).data.ptr + (size_t)(mat).step*(row)+(pix_size)*(col))

inline float
min4(float a, float b, float c, float d)
{
	a = MIN(a, b);
	c = MIN(c, d);
	return MIN(a, c);
}

#define CV_MAT_3COLOR_ELEM(img,type,y,x,c) CV_MAT_ELEM(img,type,y,(x)*3+(c))
#define KNOWN  0  //known outside narrow band
#define BAND   1  //narrow band (known)
#define INSIDE 2  //unknown
#define CHANGE 3  //servise

typedef struct CvHeapElem
{
	float T;
	int i, j;
	struct CvHeapElem* prev;
	struct CvHeapElem* next;
}
CvHeapElem;


class CvPriorityQueueFloat
{
protected:
	CvHeapElem *mem, *empty, *head, *tail;
	int num, in;

public:
	bool Init(const CvMat* f)
	{
		int i, j;
		for (i = num = 0; i < f->rows; i++)
		{
			for (j = 0; j < f->cols; j++)
				num += CV_MAT_ELEM(*f, uchar, i, j) != 0;
		}
		if (num <= 0) return false;
		mem = (CvHeapElem*)cvAlloc((num + 2)*sizeof(CvHeapElem));
		if (mem == NULL) return false;

		head = mem;
		head->i = head->j = -1;
		head->prev = NULL;
		head->next = mem + 1;
		head->T = -FLT_MAX;
		empty = mem + 1;
		for (i = 1; i <= num; i++) {
			mem[i].prev = mem + i - 1;
			mem[i].next = mem + i + 1;
			mem[i].i = -1;
			mem[i].T = FLT_MAX;
		}
		tail = mem + i;
		tail->i = tail->j = -1;
		tail->prev = mem + i - 1;
		tail->next = NULL;
		tail->T = FLT_MAX;
		return true;
	}

	bool Add(const CvMat* f) {
		int i, j;
		for (i = 0; i<f->rows; i++) {
			for (j = 0; j<f->cols; j++) {
				if (CV_MAT_ELEM(*f, uchar, i, j) != 0) {
					if (!Push(i, j, 0)) return false;
				}
			}
		}
		return true;
	}

	bool Push(int i, int j, float T) {
		CvHeapElem *tmp = empty, *add = empty;
		if (empty == tail) return false;
		while (tmp->prev->T>T) tmp = tmp->prev;
		if (tmp != empty) {
			add->prev->next = add->next;
			add->next->prev = add->prev;
			empty = add->next;
			add->prev = tmp->prev;
			add->next = tmp;
			add->prev->next = add;
			add->next->prev = add;
		}
		else {
			empty = empty->next;
		}
		add->i = i;
		add->j = j;
		add->T = T;
		in++;
		//      printf("push i %3d  j %3d  T %12.4e  in %4d\n",i,j,T,in);
		return true;
	}

	bool Pop(int *i, int *j) {
		CvHeapElem *tmp = head->next;
		if (empty == tmp) return false;
		*i = tmp->i;
		*j = tmp->j;
		tmp->prev->next = tmp->next;
		tmp->next->prev = tmp->prev;
		tmp->prev = empty->prev;
		tmp->next = empty;
		tmp->prev->next = tmp;
		tmp->next->prev = tmp;
		empty = tmp;
		in--;
		//      printf("pop  i %3d  j %3d  T %12.4e  in %4d\n",tmp->i,tmp->j,tmp->T,in);
		return true;
	}

	bool Pop(int *i, int *j, float *T) {
		CvHeapElem *tmp = head->next;
		if (empty == tmp) return false;
		*i = tmp->i;
		*j = tmp->j;
		*T = tmp->T;
		tmp->prev->next = tmp->next;
		tmp->next->prev = tmp->prev;
		tmp->prev = empty->prev;
		tmp->next = empty;
		tmp->prev->next = tmp;
		tmp->next->prev = tmp;
		empty = tmp;
		in--;
		//      printf("pop  i %3d  j %3d  T %12.4e  in %4d\n",tmp->i,tmp->j,tmp->T,in);
		return true;
	}

	CvPriorityQueueFloat(void) {
		num = in = 0;
		mem = empty = head = tail = NULL;
	}

	~CvPriorityQueueFloat(void)
	{
		cvFree(&mem);
	}
};

inline float VectorScalMult(CvPoint2D32f v1, CvPoint2D32f v2) {
	return v1.x*v2.x + v1.y*v2.y;
}

inline float VectorLength(CvPoint2D32f v1) {
	return v1.x*v1.x + v1.y*v1.y;
}

///////////////////////////////////////////////////////////////////////////////////////////
//HEAP::iterator Heap_Iterator;
//HEAP Heap;

static float FastMarching_solve(int i1, int j1, int i2, int j2, const CvMat* f, const CvMat* t)
{
	double sol, a11, a22, m12;
	a11 = CV_MAT_ELEM(*t, float, i1, j1);
	a22 = CV_MAT_ELEM(*t, float, i2, j2);
	m12 = MIN(a11, a22);

	if (CV_MAT_ELEM(*f, uchar, i1, j1) != INSIDE)
	if (CV_MAT_ELEM(*f, uchar, i2, j2) != INSIDE)
	if (fabs(a11 - a22) >= 1.0)
		sol = 1 + m12;
	else
		sol = (a11 + a22 + sqrt((double)(2 - (a11 - a22)*(a11 - a22))))*0.5;
	else
		sol = 1 + a11;
	else if (CV_MAT_ELEM(*f, uchar, i2, j2) != INSIDE)
		sol = 1 + a22;
	else
		sol = 1 + m12;

	return (float)sol;
}
/////////////////////////////////////////////////////////////////////////////////////


static void icvCalcFMM(const CvMat *f, CvMat *t, CvMat *color, CvPriorityQueueFloat *Heap, int InpaintRange, bool negate) {
	int i, j, ii = 0, jj = 0, q;
	float dist;

	while (Heap->Pop(&ii, &jj)) {

		unsigned known = (negate) ? CHANGE : KNOWN;
		CV_MAT_ELEM(*f, uchar, ii, jj) = (uchar)known;

		
		for (q = 0; q<4; q++) {
			i = ii + dxy[q][0], j = jj + dxy[q][1];
			if ((i <= 0) || (j <= 0) || (i>f->rows) || (j>f->cols)) continue;

			if (CV_MAT_ELEM(*f, uchar, i, j) == INSIDE) {
				dist = min4(FastMarching_solve(i - 1, j, i, j - 1, f, t),
					FastMarching_solve(i + 1, j, i, j - 1, f, t),
					FastMarching_solve(i - 1, j, i, j + 1, f, t),
					FastMarching_solve(i + 1, j, i, j + 1, f, t));

				float Cp = 0, colp = CV_MAT_ELEM(*color, uchar, i - 1, j - 1), colq;
				int cnt = 0, ci = i - 1, cj = j - 1;
				for (int k = ci - InpaintRange; k <= ci + InpaintRange; k++) {
					for (int l = cj - InpaintRange; l <= cj + InpaintRange; l++) {
						if (k > 0 && l > 0 && k < t->rows - 1 && l < t->cols - 1) {
							if ((CV_MAT_ELEM(*f, uchar, k, l) != INSIDE) &&
								((l - cj)*(l - cj) + (k - ci)*(k - ci) <= InpaintRange*InpaintRange)) {
								cnt++;
								colq = CV_MAT_ELEM(*color, uchar, k, l);
								Cp += exp(-(colq - colp) * (colq - colp) / (2 * sigmac * sigmac));
							}
						}
					}
				}
				Cp /= (float)cnt;
				dist = (1.0 - lambdat) * dist + lambdat * (1.0 - Cp);

				CV_MAT_ELEM(*t, float, i, j) = dist;
				CV_MAT_ELEM(*f, uchar, i, j) = BAND;
				Heap->Push(i, j, dist);
			}
		}
	}

	if (negate) {
		for (i = 0; i<f->rows; i++) {
			for (j = 0; j<f->cols; j++) {
				if (CV_MAT_ELEM(*f, uchar, i, j) == CHANGE) {
					CV_MAT_ELEM(*f, uchar, i, j) = KNOWN;
					CV_MAT_ELEM(*t, float, i, j) = -CV_MAT_ELEM(*t, float, i, j);
				}
			}
		}
	}
}


static void icvTeleaInpaintFMM(const CvMat *f, CvMat *t, CvMat *out, CvMat *color_img, int range, CvPriorityQueueFloat *Heap) {
	int i = 0, j = 0, ii = 0, jj = 0, k, l, q, color = 0;
	float dist;

	//freopen("out.txt", "r", stdout);

	if (CV_MAT_CN(out->type) == 1) {

		while (Heap->Pop(&ii, &jj)) {

			CV_MAT_ELEM(*f, uchar, ii, jj) = KNOWN;
			for (q = 0; q<4; q++) {
				i = ii + dxy[q][0], j = jj + dxy[q][1];
				if ((i <= 1) || (j <= 1) || (i>t->rows - 1) || (j>t->cols - 1)) continue;

				if (CV_MAT_ELEM(*f, uchar, i, j) == INSIDE) {
					dist = min4(FastMarching_solve(i - 1, j, i, j - 1, f, t),
						FastMarching_solve(i + 1, j, i, j - 1, f, t),
						FastMarching_solve(i - 1, j, i, j + 1, f, t),
						FastMarching_solve(i + 1, j, i, j + 1, f, t));


					float Cp = 0, colp = CV_MAT_ELEM(*color_img, uchar, i - 1, j - 1), colq;
					int cnt = 0, ci = i - 1, cj = j - 1;
					for (int k = ci - range; k <= ci + range; k++) {
						for (int l = cj - range; l <= cj + range; l++) {
							if (k > 0 && l > 0 && k < t->rows - 1 && l < t->cols - 1) {
								if ((CV_MAT_ELEM(*f, uchar, k, l) != INSIDE) &&
									((l - cj)*(l - cj) + (k - ci)*(k - ci) <= range*range)) {
									cnt++;
									colq = CV_MAT_ELEM(*color_img, uchar, k, l);
									Cp += exp(-(colq - colp) * (colq - colp) / (2 * sigmac * sigmac));
								}
							}
						}
					}
					Cp /= (float)cnt;
					dist = (1.0 - lambda) * dist + lambda * (1.0 - Cp);
					CV_MAT_ELEM(*t, float, i, j) = dist;


					CvPoint2D32f gradI, gradT, r;
					float Ia = 0, Jx = 0, Jy = 0, s = 1.0e-20f, w, dst, lev, dir, sat, col;

					if (CV_MAT_ELEM(*f, uchar, i, j + 1) != INSIDE) {
						if (CV_MAT_ELEM(*f, uchar, i, j - 1) != INSIDE) {
							gradT.x = (float)((CV_MAT_ELEM(*t, float, i, j + 1) - CV_MAT_ELEM(*t, float, i, j - 1)))*0.5f;
						}
						else {
							gradT.x = (float)((CV_MAT_ELEM(*t, float, i, j + 1) - CV_MAT_ELEM(*t, float, i, j)));
						}
					}
					else {
						if (CV_MAT_ELEM(*f, uchar, i, j - 1) != INSIDE) {
							gradT.x = (float)((CV_MAT_ELEM(*t, float, i, j) - CV_MAT_ELEM(*t, float, i, j - 1)));
						}
						else {
							gradT.x = 0;
						}
					}
					if (CV_MAT_ELEM(*f, uchar, i + 1, j) != INSIDE) {
						if (CV_MAT_ELEM(*f, uchar, i - 1, j) != INSIDE) {
							gradT.y = (float)((CV_MAT_ELEM(*t, float, i + 1, j) - CV_MAT_ELEM(*t, float, i - 1, j)))*0.5f;
						}
						else {
							gradT.y = (float)((CV_MAT_ELEM(*t, float, i + 1, j) - CV_MAT_ELEM(*t, float, i, j)));
						}
					}
					else {
						if (CV_MAT_ELEM(*f, uchar, i - 1, j) != INSIDE) {
							gradT.y = (float)((CV_MAT_ELEM(*t, float, i, j) - CV_MAT_ELEM(*t, float, i - 1, j)));
						}
						else {
							gradT.y = 0;
						}
					}
					for (k = i - range; k <= i + range; k++) {
						int km = k - 1 + (k == 1), kp = k - 1 - (k == t->rows - 2);
						for (l = j - range; l <= j + range; l++) {
							int lm = l - 1 + (l == 1), lp = l - 1 - (l == t->cols - 2);
							if (k > 0 && l > 0 && k < t->rows - 1 && l < t->cols - 1) {
								if ((CV_MAT_ELEM(*f, uchar, k, l) != INSIDE) &&
									((l - j)*(l - j) + (k - i)*(k - i) <= range*range)) {
									r.y = (float)(i - k);
									r.x = (float)(j - l);

									dst = (float)(1. / (VectorLength(r)*sqrt(VectorLength(r))));
									lev = (float)(1. / (1 + fabs(CV_MAT_ELEM(*t, float, k, l) - CV_MAT_ELEM(*t, float, i, j))));

									dir = VectorScalMult(r, gradT);
									if (fabs(dir) <= 0.01) dir = 0.000001f;

									float colp = CV_MAT_ELEM(*color_img, uchar, i - 1, j - 1), colq = CV_MAT_ELEM(*color_img, uchar, km, lm);
									col = exp(-(colp - colq) * (colp - colq) / (2 * sigmac * sigmac));
									col = 1;
									dir = 1;
									lev = 1;
									dst = 1;
									w = (float)fabs(dst*lev*dir*col);
									//std::cout << dst << ' ' << lev << ' ' << dir << ' ' << col << endl;

									if (CV_MAT_ELEM(*f, uchar, k, l + 1) != INSIDE) {
										if (CV_MAT_ELEM(*f, uchar, k, l - 1) != INSIDE) {
											gradI.x = (float)((CV_MAT_ELEM(*out, uchar, km, lp + 1) - CV_MAT_ELEM(*out, uchar, km, lm - 1)))*2.0f;
										}
										else {
											gradI.x = (float)((CV_MAT_ELEM(*out, uchar, km, lp + 1) - CV_MAT_ELEM(*out, uchar, km, lm)));
										}
									}
									else {
										if (CV_MAT_ELEM(*f, uchar, k, l - 1) != INSIDE) {
											gradI.x = (float)((CV_MAT_ELEM(*out, uchar, km, lp) - CV_MAT_ELEM(*out, uchar, km, lm - 1)));
										}
										else {
											gradI.x = 0;
										}
									}
									if (CV_MAT_ELEM(*f, uchar, k + 1, l) != INSIDE) {
										if (CV_MAT_ELEM(*f, uchar, k - 1, l) != INSIDE) {
											gradI.y = (float)((CV_MAT_ELEM(*out, uchar, kp + 1, lm) - CV_MAT_ELEM(*out, uchar, km - 1, lm)))*2.0f;
										}
										else {
											gradI.y = (float)((CV_MAT_ELEM(*out, uchar, kp + 1, lm) - CV_MAT_ELEM(*out, uchar, km, lm)));
										}
									}
									else {
										if (CV_MAT_ELEM(*f, uchar, k - 1, l) != INSIDE) {
											gradI.y = (float)((CV_MAT_ELEM(*out, uchar, kp, lm) - CV_MAT_ELEM(*out, uchar, km - 1, lm)));
										}
										else {
											gradI.y = 0;
										}
									}
									Ia += (float)w * (float)(CV_MAT_ELEM(*out, uchar, km, lm));
									/*Jx -= (float)w * (float)(gradI.x*r.x);
									Jy -= (float)w * (float)(gradI.y*r.y);*/
									s += w;
								}
							}
						}
					}
					//sat = (float)((Ia / s + (Jx + Jy) / (sqrt(Jx*Jx + Jy*Jy) + 1.0e-20f) + 0.5f));
					sat = (float)(Ia / s);
					{
						CV_MAT_ELEM(*out, uchar, i - 1, j - 1) = cv::saturate_cast<uchar>(sat);
					}

					CV_MAT_ELEM(*f, uchar, i, j) = BAND;
					Heap->Push(i, j, dist);
				}
			
			}
		}
	}
}


void cvInpaint(const CvArr* _input_img, const CvArr* _input_img_color, const CvArr* _inpaint_mask, CvArr* _output_img, double inpaintRange)
{
	cv::Ptr<CvMat> mask, band, f, t, out;
	cv::Ptr<CvPriorityQueueFloat> Heap, Out;
	cv::Ptr<IplConvKernel> el_cross, el_range;

	CvMat input_hdr, mask_hdr, output_hdr, input_color_hdr;
	CvMat* input_img, *inpaint_mask, *output_img, *input_img_color;
	int range = cvRound(inpaintRange);
	int erows, ecols;

	input_img = cvGetMat(_input_img, &input_hdr);
	input_img_color = cvGetMat(_input_img_color, &input_color_hdr);
	inpaint_mask = cvGetMat(_inpaint_mask, &mask_hdr);
	output_img = cvGetMat(_output_img, &output_hdr);


	if (!CV_ARE_SIZES_EQ(input_img, output_img) || !CV_ARE_SIZES_EQ(input_img, inpaint_mask))
		CV_Error(CV_StsUnmatchedSizes, "All the input and output images must have the same size");

	if ((CV_MAT_TYPE(input_img->type) != CV_8UC1 &&
		CV_MAT_TYPE(input_img->type) != CV_8UC3) ||
		!CV_ARE_TYPES_EQ(input_img, output_img))
		CV_Error(CV_StsUnsupportedFormat,
		"Only 8-bit 1-channel and 3-channel input/output images are supported");

	if (CV_MAT_TYPE(inpaint_mask->type) != CV_8UC1)
		CV_Error(CV_StsUnsupportedFormat, "The mask must be 8-bit 1-channel image");

	range = MAX(range, 1);
	range = MIN(range, 100);

	ecols = input_img->cols + 2;
	erows = input_img->rows + 2;

	f = cvCreateMat(erows, ecols, CV_8UC1);
	t = cvCreateMat(erows, ecols, CV_32FC1);
	band = cvCreateMat(erows, ecols, CV_8UC1);
	mask = cvCreateMat(erows, ecols, CV_8UC1);
	el_cross = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_CROSS, NULL);

	cvCopy(input_img, output_img);
	cvSet(mask, cvScalar(KNOWN, 0, 0, 0));
	//COPY_MASK_BORDER1_C1(inpaint_mask, mask, uchar);
	for (int i = 0; i < inpaint_mask->rows; i++) 
	for (int j = 0; j < inpaint_mask->cols; j++)
	if (CV_MAT_ELEM(*inpaint_mask, uchar, i, j) != 0)
		CV_MAT_ELEM(*mask, uchar, i + 1, j + 1) = INSIDE;

	//SET_BORDER1_C1(mask, uchar, 0);
	for (int i = 0; i < mask->cols; i++) CV_MAT_ELEM(*mask, uchar, 0, i) = 0;
	for (int i = 1; i < mask->rows - 1; i++) CV_MAT_ELEM(*mask, uchar, i, 0) = CV_MAT_ELEM(*mask, uchar, i, mask->cols - 1) = 0;
	for (int i = 0; i < mask->cols; i++) CV_MAT_ELEM(*mask, uchar, erows - 1, i) = 0;

	cvSet(f, cvScalar(KNOWN, 0, 0, 0));
	cvSet(t, cvScalar(1.0e6f, 0, 0, 0));
	cvDilate(mask, band, el_cross, 1);   // image with narrow band
	Heap = new CvPriorityQueueFloat;
	if (!Heap->Init(band))
		return;
	cvSub(band, mask, band, NULL);
	//SET_BORDER1_C1(band, uchar, 0);
	for (int i = 0; i < band->cols; i++) CV_MAT_ELEM(*band, uchar, 0, i) = 0;
	for (int i = 1; i < band->rows - 1; i++) CV_MAT_ELEM(*band, uchar, i, 0) = CV_MAT_ELEM(*band, uchar, i, band->cols - 1) = 0;
	for (int i = 0; i < band->cols; i++) CV_MAT_ELEM(*band, uchar, erows - 1, i) = 0;
	if (!Heap->Add(band))
		return;
	cvSet(f, cvScalar(BAND, 0, 0, 0), band);
	cvSet(f, cvScalar(INSIDE, 0, 0, 0), mask);
	cvSet(t, cvScalar(0, 0, 0, 0), band);


	out = cvCreateMat(erows, ecols, CV_8UC1);
	el_range = cvCreateStructuringElementEx(2 * range + 1, 2 * range + 1,
		range, range, CV_SHAPE_ELLIPSE, NULL);
	cvDilate(mask, out, el_range, 1);
	cvSub(out, mask, out, NULL);
	Out = new CvPriorityQueueFloat;
	if (!Out->Init(out))
		return;
	if (!Out->Add(band))
		return;
	cvSub(out, band, out, NULL);
	//SET_BORDER1_C1(out, uchar, 0);
	for (int i = 0; i < out->cols; i++) CV_MAT_ELEM(*out, uchar, 0, i) = 0;
	for (int i = 1; i < out->rows - 1; i++) CV_MAT_ELEM(*out, uchar, i, 0) = CV_MAT_ELEM(*out, uchar, i, out->cols - 1) = 0;
	for (int i = 0; i < out->cols; i++) CV_MAT_ELEM(*out, uchar, erows - 1, i) = 0;
	icvCalcFMM(out, t, input_img_color, Out, inpaintRange, true);
	icvTeleaInpaintFMM(mask, t, output_img, input_img_color, range, Heap);
}

void inpaint(InputArray _src, InputArray _color, InputArray _mask, OutputArray _dst, double inpaintRange)
{
	Mat src = _src.getMat(), mask = _mask.getMat(), color = _color.getMat();
	_dst.create(src.size(), src.type());
	CvMat c_src = src, c_mask = mask, c_dst = _dst.getMat(), c_color = color;
	cvInpaint(&c_src, &c_color, &c_mask, &c_dst, inpaintRange);
}

Mat vis;
stack<Point> sta;
queue<Point> Q;

void Change(Point p, Mat img) {
	bool flag = true;
	while (!Q.empty()) Q.pop();
	while (!sta.empty()) sta.pop();
	Q.push(p);
	sta.push(p);
	vis.at<uchar>(p.y, p.x) = 1;
	while (!Q.empty()) {
		Point now = Q.front(); Q.pop();
		for (int i = 0; i < 4; i++) {
			Point q = Point(now.x + dxy[i][0], now.y + dxy[i][1]);
			if (q.x < 0 || q.x >= img.cols || q.y < 0 || q.y >= img.rows) continue;
			if (vis.at<uchar>(q.y, q.x) != 0) continue;
			if (img.at<uchar>(q.y, q.x) != 0) continue;
			if (q.x == 0 || q.x == img.cols - 1 || q.y == 0 || q.y == img.rows - 1){
				flag = false;
				continue;
			}
			Q.push(q);
			sta.push(q);
			vis.at<uchar>(q.y, q.x) = 1;
		}
	}
	if (!flag) return;
	while (!sta.empty()) {
		Point top = sta.top();  sta.pop();
		line(inpaintMask, top, top, Scalar::all(255), 1, 4, 0);
	}
}
void getMask(Mat img) {
	vis = Mat::zeros(img.size(), CV_8U);
	for (int i = 1; i < img.cols - 1; i++) {
		for (int j = 1; j < img.rows - 1; j++) {
			if (img.at<uchar>(j, i) == 0 && vis.at<uchar>(j, i) == 0){
				Change(Point(i, j), img);
			}
		}
	}

	/*for (int i = 0; i < img.cols; i++)
	for (int j = 0; j < img.rows; j++)
	if (img.at<uchar>(j, i) == 0)
		line(inpaintMask, Point(i, j), Point(i, j), Scalar::all(255), 1, 8, 0);*/
}

vector<string> rgbID;
int main(int argc, char** argv)
{

	ifstream inKeyFrame("kayFrame.txt");
	ifstream inrgb("rgb.txt");
	rgbID.clear();
	while (!inrgb.eof()) {
		string id, path;
		inrgb >> id >> path;
		if (id == "") break;
		rgbID.push_back(id);
	}
	inrgb.close();
	while (!inKeyFrame.eof()) {
		string depthname;
		int id;
		inKeyFrame >> depthname >> id;
		if (depthname == "") break;
		cout << rgbID[id] << ' ' << depthname << endl;

		string fold = "keyframe\\", foldrgb = "rgb\\", png = ".png";
		Mat imgdp0 = imread(fold + depthname, -1);
		Mat imgcolor = imread(foldrgb + rgbID[id] + png, -1);
		if (imgdp0.empty()){
			cout << "Couldn't open the image " << depthname << ". Usage: inpaint <image_name>\n" << endl;
			return 0;
		}
		if (imgcolor.empty()){
			cout << "Couldn't open the image " << rgbID[id] << ". Usage: inpaint <image_name>\n" << endl;
			return 0;
		}

		//namedWindow("image", 1);

		img = imgdp0.clone();
		inpaintMask = Mat::zeros(img.size(), CV_8U);

		//imshow("image", img);
	
		Mat inpainted;

		getMask(img);
 		//imshow("mask", inpaintMask);
		Mat inpaintMask2;
		GaussianBlur(inpaintMask, inpaintMask2, Size(5, 5), 0, 0);
		//imshow("mask2", inpaintMask);
		inpaint(img, imgcolor, inpaintMask2, inpainted, 3);//, CV_INPAINT_TELEA);
		//imshow("inpainted image", inpainted);
		imwrite("inapinted//inapint_" + depthname, inpainted);

	}
	inKeyFrame.close();
	return 0;
}
