#include "decision.h"

void hg_decider_init(hg_decider_t *d, float th, int need)
{
    d->th     = th;
    d->need   = (need < 1) ? 1 : need;
    d->streak = 0;
    d->cls    = -1;
    d->fired  = 0;
}

int hg_decider_push(hg_decider_t *d, int cls, float prob)
{
    if (prob < d->th) {
        /* 置信度不足：计数清零。注意不能用 continue，必须连同类别一起复位，
         * 否则下一次同类事件会接着上一轮的计数继续累加。 */
        d->streak = 0;
        d->cls    = -1;
        d->fired  = 0;
        return 0;
    }

    if (cls != d->cls) {
        /* 换了一类：重新从 1 开始计数，避免不同类别互相凑数 */
        d->cls    = cls;
        d->streak = 1;
        d->fired  = 0;
    } else {
        d->streak++;
    }

    /* 只在「刚好凑满 need 次」的那一帧触发一次。
     * 之后 streak 继续增长，不会再等于 need，因此不会重复报警。 */
    if (d->streak == d->need && !d->fired) {
        d->fired = 1;
        return 1;
    }
    return 0;
}
