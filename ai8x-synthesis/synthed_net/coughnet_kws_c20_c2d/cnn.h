/**************************************************************************************************
* Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically @generated for the coughnet_kws_c20_c2d network from a template.
 * Please do not edit; instead, edit the template and regenerate.
 */

#ifndef __CNN_H__
#define __CNN_H__

#include <stdint.h>
typedef int32_t q31_t;
typedef int16_t q15_t;

/* Return codes */
#define CNN_FAIL 0
#define CNN_OK 1

/*
  SUMMARY OF OPS
  Hardware: 13,859,456 ops (13,800,192 macc; 59,264 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0: 1,651,200 ops (1,638,400 macc; 12,800 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 1,292,800 ops (1,280,000 macc; 12,800 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 646,400 ops (640,000 macc; 6,400 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 104,448 ops (102,400 macc; 2,048 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 593,920 ops (589,824 macc; 4,096 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 2,367,488 ops (2,359,296 macc; 8,192 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 4,726,784 ops (4,718,592 macc; 8,192 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 2,215,680 ops (2,211,840 macc; 3,840 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 242,816 ops (241,920 macc; 896 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9: 17,920 ops (17,920 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 125,594 bytes out of 442,368 bytes total (28.4%)
  Bias memory:   0 bytes out of 2,048 bytes total (0.0%)
*/

/* Number of outputs for this network */
#define CNN_NUM_OUTPUTS 20

/* Use this timer to time the inference */
#define CNN_INFERENCE_TIMER MXC_TMR0

/* Port pin actions used to signal that processing is active */

#define CNN_START LED_On(1)
#define CNN_COMPLETE LED_Off(1)
#define SYS_START LED_On(0)
#define SYS_COMPLETE LED_Off(0)

/* Run software SoftMax on unloaded data */
void softmax_q17p14_q15(const q31_t * vec_in, const uint16_t dim_vec, q15_t * p_out);
/* Shift the input, then calculate SoftMax */
void softmax_shift_q17p14_q15(q31_t * vec_in, const uint16_t dim_vec, uint8_t in_shift, q15_t * p_out);

/* Stopwatch - holds the runtime when accelerator finishes */
extern volatile uint32_t cnn_time;

/* Custom memcopy routines used for weights and data */
void memcpy32(uint32_t *dst, const uint32_t *src, int n);
void memcpy32_const(uint32_t *dst, int n);

/* Enable clocks and power to accelerator, enable interrupt */
int cnn_enable(uint32_t clock_source, uint32_t clock_divider);

/* Disable clocks and power to accelerator */
int cnn_disable(void);

/* Perform minimum accelerator initialization so it can be configured */
int cnn_init(void);

/* Configure accelerator for the given network */
int cnn_configure(void);

/* Load accelerator weights */
int cnn_load_weights(void);

/* Verify accelerator weights (debug only) */
int cnn_verify_weights(void);

/* Load accelerator bias values (if needed) */
int cnn_load_bias(void);

/* Start accelerator processing */
int cnn_start(void);

/* Force stop accelerator */
int cnn_stop(void);

/* Continue accelerator after stop */
int cnn_continue(void);

/* Unload results from accelerator */
int cnn_unload(uint32_t *out_buf);

/* Turn on the boost circuit */
int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin);

/* Turn off the boost circuit */
int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin);

#endif // __CNN_H__
