#ifndef HG_DECISION_H
#define HG_DECISION_H

/* 同 mfcc.h：由 C 实现，被 C++ 引用，必须加 extern "C" 否则链接期找不到符号。 */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * 报警决策：置信度阈值 + 同一事件连续 N 次确认
 * ============================================================
 * 单次推理超过阈值就直接报警，误报会非常频繁：相邻两个推理窗口
 * 有 32 帧中的 31 帧重叠，一次偶发误判很容易被后续窗口重复触发。
 *
 * 这里要求「连续 N 个推理窗口判为同一类、且置信度都超过阈值」才触发。
 * 之所以要判「同一类」：如果只看置信度不分类别，玻璃碎裂、婴儿哭声、
 * 敲击三种误判可以轮流凑数，三次就报警，等于门槛形同虚设。
 *
 * 触发采用边沿方式：报过一次后持续超阈值不会重复报警，
 * 必须等置信度掉回阈值以下重新计数，才可能再次触发，避免刷屏。
 */

typedef struct {
    float th;      /* 置信度阈值，例如 0.85 */
    int   need;    /* 需要连续确认的次数，例如 3 */
    int   streak;  /* 当前已连续确认的次数 */
    int   cls;     /* 正在累计的类别；-1 表示无 */
    int   fired;   /* 本轮是否已报警，掉回阈值以下才复位 */
} hg_decider_t;

void hg_decider_init(hg_decider_t *d, float th, int need);

/*
 * 送入一次推理结果。
 * cls  : 置信度最高的类别编号
 * prob : 该类别的置信度（0~1）
 * 返回 1 表示本次触发报警。
 */
int hg_decider_push(hg_decider_t *d, int cls, float prob);

#ifdef __cplusplus
}
#endif

#endif /* HG_DECISION_H */
