/* WDBC feature selection: HC, TS, SA, and GA under shared evaluation rules. */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NF 30
#define EVAL_BUDGET 3000UL
#define POP_SIZE 30
#define TOURNAMENT_SIZE 3
#define TABU_TENURE 7

typedef struct {
  double x[NF];
  int y;
} Sample;
typedef struct {
  Sample *a;
  size_t n;
} Data;
typedef struct {
  unsigned long long s;
} Rng;
typedef struct {
  unsigned char b[NF];
} Sol;
typedef struct {
  int folds;
  int epochs;
  double lr;
  double penalty;
  int *fold_id;
} Config;
typedef struct {
  Sol best;
  double fit;
  unsigned long evals;
} Result;
typedef struct {
  double mean[NF];
  double sd[NF];
  double w[NF + 1];
} Model;

static unsigned rnd(Rng *r) {
  r->s ^= r->s << 7;
  r->s ^= r->s >> 9;
  return (unsigned)(r->s >> 16);
}
static double unit(Rng *r) { return rnd(r) / ((double)UINT_MAX + 1.0); }
static int ri(Rng *r, int n) { return (int)(unit(r) * n); }
static void shuffle(int *a, int n, Rng *r) {
  for (int i = n - 1; i > 0; i--) {
    int j = ri(r, i + 1);
    int t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
}
static int count(const Sol *s) {
  int n = 0;
  for (int i = 0; i < NF; i++)
    n += s->b[i];
  return n;
}
static void random_sol(Sol *s, Rng *r) {
  do {
    for (int i = 0; i < NF; i++)
      s->b[i] = (unsigned char)ri(r, 2);
  } while (count(s) == 0);
}

/* Make a one-bit neighbor.  An all-zero subset is invalid. */
static int make_neighbor(const Sol *src, Sol *dst, int bit) {
  *dst = *src;
  dst->b[bit] ^= 1;
  return count(dst) != 0;
}
static int valid_neighbor_count(const Sol *s) {
  return NF - (count(s) == 1 ? 1 : 0);
}

static int load_csv(const char *path, Data *d) {
  FILE *fp = fopen(path, "r");
  char line[8192];
  size_t cap = 0;
  d->a = NULL;
  d->n = 0;
  if (!fp || !fgets(line, sizeof line, fp)) {
    if (fp)
      fclose(fp);
    return 0;
  }
  while (fgets(line, sizeof line, fp)) {
    Sample z;
    char *p = line;
    char *e;
    (void)strtod(p, &e);
    if (e == p)
      continue;
    p = *e == ',' ? e + 1 : e;
    z.y = *p == 'M';
    while (*p && *p != ',')
      p++;
    if (*p == ',')
      p++;
    for (int j = 0; j < NF; j++) {
      z.x[j] = strtod(p, &e);
      if (e == p) {
        fclose(fp);
        free(d->a);
        d->a = NULL;
        d->n = 0;
        return 0;
      }
      p = *e == ',' ? e + 1 : e;
    }
    if (d->n == cap) {
      size_t next = cap ? cap * 2 : 128;
      Sample *q = realloc(d->a, next * sizeof *q);
      if (!q) {
        fclose(fp);
        free(d->a);
        d->a = NULL;
        d->n = 0;
        return 0;
      }
      d->a = q;
      cap = next;
    }
    d->a[d->n++] = z;
  }
  fclose(fp);
  return d->n > 0;
}

/* Stratified 80/20 split using dynamically sized class-index arrays. */
static int split(const Data *all, Data *tr, Data *te, Rng *r) {
  int *pos = NULL;
  int *neg = NULL;
  size_t np = 0, nn = 0, nte, ntr;
  tr->a = te->a = NULL;
  tr->n = te->n = 0;
  pos = malloc(all->n * sizeof *pos);
  neg = malloc(all->n * sizeof *neg);
  if (!pos || !neg)
    goto fail;
  for (size_t i = 0; i < all->n; i++) {
    if (all->a[i].y)
      pos[np++] = (int)i;
    else
      neg[nn++] = (int)i;
  }
  shuffle(pos, (int)np, r);
  shuffle(neg, (int)nn, r);
  nte = np / 5 + nn / 5;
  ntr = all->n - nte;
  tr->a = malloc(ntr * sizeof *tr->a);
  te->a = malloc(nte * sizeof *te->a);
  if (!tr->a || !te->a)
    goto fail;
  for (size_t i = 0; i < np; i++)
    (i < np / 5 ? te->a : tr->a)[i < np / 5 ? te->n++ : tr->n++] =
        all->a[pos[i]];
  for (size_t i = 0; i < nn; i++)
    (i < nn / 5 ? te->a : tr->a)[i < nn / 5 ? te->n++ : tr->n++] =
        all->a[neg[i]];
  free(pos);
  free(neg);
  return 1;
fail:
  free(pos);
  free(neg);
  free(tr->a);
  free(te->a);
  tr->a = te->a = NULL;
  tr->n = te->n = 0;
  return 0;
}

/* Fixed, stratified fold assignment independent of every search RNG. */
static int make_folds(const Data *d, int folds, int **out) {
  int *id = malloc(d->n * sizeof *id);
  int pos_fold = 0, neg_fold = 0;
  if (!id)
    return 0;
  for (size_t i = 0; i < d->n; i++) {
    if (d->a[i].y)
      id[i] = pos_fold++ % folds;
    else
      id[i] = neg_fold++ % folds;
  }
  *out = id;
  return 1;
}

static double sig(double z) {
  if (z > 35.0)
    return 1.0;
  if (z < -35.0)
    return 0.0;
  return 1.0 / (1.0 + exp(-z));
}

/* Standardisation uses only the supplied training indices. */
static void fit(Model *m, const Data *d, const int *idx, int n, const Sol *s,
                const Config *c) {
  memset(m, 0, sizeof *m);
  for (int i = 0; i < n; i++) {
    const Sample *z = &d->a[idx[i]];
    for (int j = 0; j < NF; j++)
      if (s->b[j])
        m->mean[j] += z->x[j];
  }
  for (int j = 0; j < NF; j++)
    if (s->b[j])
      m->mean[j] /= n;
  for (int i = 0; i < n; i++) {
    const Sample *z = &d->a[idx[i]];
    for (int j = 0; j < NF; j++)
      if (s->b[j]) {
        double q = z->x[j] - m->mean[j];
        m->sd[j] += q * q;
      }
  }
  for (int j = 0; j < NF; j++)
    if (s->b[j]) {
      m->sd[j] = sqrt(m->sd[j] / n);
      if (m->sd[j] < 1e-12)
        m->sd[j] = 1.0;
    }

  /* Unregularized logistic regression trained by batch gradient descent. */
  for (int e = 0; e < c->epochs; e++) {
    double grad[NF] = {0};
    double bias_grad = 0.0;
    for (int i = 0; i < n; i++) {
      const Sample *z = &d->a[idx[i]];
      double v = m->w[NF];
      for (int j = 0; j < NF; j++)
        if (s->b[j])
          v += m->w[j] * (z->x[j] - m->mean[j]) / m->sd[j];
      v = z->y - sig(v);
      for (int j = 0; j < NF; j++)
        if (s->b[j])
          grad[j] += v * (z->x[j] - m->mean[j]) / m->sd[j];
      bias_grad += v;
    }
    for (int j = 0; j < NF; j++)
      if (s->b[j])
        m->w[j] += c->lr * grad[j] / n;
    m->w[NF] += c->lr * bias_grad / n;
  }
}

static int pred(const Model *m, const Sample *z, const Sol *s) {
  double v = m->w[NF];
  for (int j = 0; j < NF; j++)
    if (s->b[j])
      v += m->w[j] * (z->x[j] - m->mean[j]) / m->sd[j];
  return sig(v) >= 0.5;
}
static double f1(int tp, int fp, int fn) {
  int denominator = 2 * tp + fp + fn;
  if (denominator == 0)
    return 0.0;
  return 2.0 * tp / denominator;
}

/* Fitness uses the single fixed stratified fold assignment in Config. */
static double fitness(const Data *d, const Sol *s, const Config *c) {
  int *train = malloc(d->n * sizeof *train);
  double sum = 0.0;
  if (!train)
    return -1e99;
  for (int k = 0; k < c->folds; k++) {
    Model m;
    int n = 0, tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < d->n; i++)
      if (c->fold_id[i] != k)
        train[n++] = (int)i;
    fit(&m, d, train, n, s, c);
    for (size_t i = 0; i < d->n; i++)
      if (c->fold_id[i] == k) {
        int p = pred(&m, &d->a[i], s);
        int y = d->a[i].y;
        if (p && y)
          tp++;
        else if (p)
          fp++;
        else if (y)
          fn++;
      }
    sum += f1(tp, fp, fn);
  }
  free(train);
  return sum / c->folds - c->penalty * (double)count(s) / NF;
}
static double score(const Data *d, const Sol *s, const Config *c, Result *r) {
  r->evals++;
  return fitness(d, s, c);
}
static Result initial(const Data *d, const Config *c, Rng *r) {
  Result x = {0};
  random_sol(&x.best, r);
  x.fit = score(d, &x.best, c, &x);
  return x;
}

/* Best-improvement HC: stop at a local optimum; never restart. */
static Result hc(const Data *d, const Config *c, Rng *r) {
  Result out = initial(d, c, r);
  Result cur = out;
  for (;;) {
    int required = valid_neighbor_count(&cur.best);
    Sol iteration_best = cur.best;
    double iteration_fit = cur.fit;
    if (out.evals + (unsigned long)required > EVAL_BUDGET)
      break;
    for (int bit = 0; bit < NF; bit++) {
      Sol z;
      if (make_neighbor(&cur.best, &z, bit)) {
        double q = score(d, &z, c, &out);
        if (q > iteration_fit) {
          iteration_fit = q;
          iteration_best = z;
        }
      }
    }
    if (iteration_fit <= cur.fit)
      break;
    cur.best = iteration_best;
    cur.fit = iteration_fit;
    if (cur.fit > out.fit) {
      out.best = cur.best;
      out.fit = cur.fit;
    }
  }
  return out;
}

/* Best-improvement TS with aspiration and complete valid neighborhoods. */
static Result ts(const Data *d, const Config *c, Rng *r) {
  Result out = initial(d, c, r);
  Result cur = out;
  int tabu[NF] = {0};
  for (int it = 0;; it++) {
    int required = valid_neighbor_count(&cur.best);
    Sol iteration_best = cur.best;
    double iteration_fit = -1e99;
    int found = 0;
    int pick = -1;
    if (out.evals + (unsigned long)required > EVAL_BUDGET)
      break;
    for (int bit = 0; bit < NF; bit++) {
      Sol z;
      if (make_neighbor(&cur.best, &z, bit)) {
        double q = score(d, &z, c, &out);
        int aspiration = q > out.fit;
        if ((tabu[bit] <= it || aspiration) &&
            (!found || q > iteration_fit)) {
          found = 1;
          pick = bit;
          iteration_fit = q;
          iteration_best = z;
        }
      }
    }
    if (!found)
      break;
    cur.best = iteration_best;
    cur.fit = iteration_fit;
    /* A move at it blocks it+1 through it+7; it+8 is permitted. */
    tabu[pick] = it + TABU_TENURE + 1;
    if (cur.fit > out.fit) {
      out.best = cur.best;
      out.fit = cur.fit;
    }
  }
  return out;
}

static Result sa(const Data *d, const Config *c, Rng *r) {
  const double initial_temperature = 0.1;
  const double minimum_temperature = 0.0001;
  const double alpha =
      pow(minimum_temperature / initial_temperature, 1.0 / (EVAL_BUDGET - 1));
  Result out = initial(d, c, r);
  Result cur = out;
  double temperature = initial_temperature;
  while (out.evals < EVAL_BUDGET) {
    Sol z;
    int bit;
    do {
      bit = ri(r, NF);
    } while (!make_neighbor(&cur.best, &z, bit));
    {
      double q = score(d, &z, c, &out);
      if (q >= cur.fit ||
          unit(r) < exp((q - cur.fit) / temperature)) {
        cur.best = z;
        cur.fit = q;
      }
    }
    if (cur.fit > out.fit) {
      out.best = cur.best;
      out.fit = cur.fit;
    }
    temperature *= alpha;
    if (temperature < minimum_temperature)
      temperature = minimum_temperature;
  }
  return out;
}

static int tournament(const double *fit, Rng *r) {
  int winner = ri(r, POP_SIZE);
  for (int i = 1; i < TOURNAMENT_SIZE; i++) {
    int candidate = ri(r, POP_SIZE);
    if (fit[candidate] > fit[winner])
      winner = candidate;
  }
  return winner;
}
static void mutate(Sol *s, Rng *r) {
  for (int j = 0; j < NF; j++)
    if (unit(r) < 1.0 / NF)
      s->b[j] ^= 1;
  if (count(s) == 0)
    s->b[ri(r, NF)] = 1;
}
static Result ga(const Data *d, const Config *c, Rng *r) {
  Sol a[POP_SIZE], b[POP_SIZE];
  double fit[POP_SIZE];
  Result out = {0};
  for (int i = 0; i < POP_SIZE; i++) {
    random_sol(&a[i], r);
    fit[i] = score(d, &a[i], c, &out);
    if (i == 0 || fit[i] > out.fit) {
      out.best = a[i];
      out.fit = fit[i];
    }
  }
  while (out.evals + POP_SIZE <= EVAL_BUDGET) {
    b[0] = out.best; /* One elite: the unchanged search-wide best. */
    for (int i = 1; i < POP_SIZE; i++) {
      int p = tournament(fit, r);
      int q = tournament(fit, r);
      if (unit(r) < 0.8) {
        int cut = 1 + ri(r, NF - 1);
        for (int j = 0; j < NF; j++)
          b[i].b[j] = j < cut ? a[p].b[j] : a[q].b[j];
      } else {
        b[i] = a[p];
      }
      mutate(&b[i], r);
    }
    memcpy(a, b, sizeof a);
    for (int i = 0; i < POP_SIZE; i++) {
      fit[i] = score(d, &a[i], c, &out);
      if (fit[i] > out.fit) {
        out.best = a[i];
        out.fit = fit[i];
      }
    }
  }
  return out;
}

static int show(const char *name, Result x, const Data *tr, const Data *te,
                const Config *c, double sec) {
  int *id = malloc(tr->n * sizeof *id);
  int tp = 0, tn = 0, fp = 0, fn = 0;
  Model m;
  if (!id)
    return 0;
  for (size_t i = 0; i < tr->n; i++)
    id[i] = (int)i;
  fit(&m, tr, id, (int)tr->n, &x.best, c);
  free(id);
  for (size_t i = 0; i < te->n; i++) {
    int p = pred(&m, &te->a[i], &x.best);
    int y = te->a[i].y;
    if (p && y)
      tp++;
    else if (p)
      fp++;
    else if (y)
      fn++;
    else
      tn++;
  }
  printf("%-2s fitness=%.4f  accuracy=%.4f  precision=%.4f  recall=%.4f  "
         "f1=%.4f  features=%d  time=%.2fs  evaluations=%lu\n",
         name, x.fit, (double)(tp + tn) / te->n,
         (tp + fp) ? (double)tp / (tp + fp) : 0.0,
         (tp + fn) ? (double)tp / (tp + fn) : 0.0, f1(tp, fp, fn),
         count(&x.best), sec, x.evals);
  printf("   mask=");
  for (int i = 0; i < NF; i++)
    putchar(x.best.b[i] ? '1' : '0');
  putchar('\n');
  return 1;
}

int main(int ac, char **av) {
  Data all = {0}, tr = {0}, te = {0};
  const char *path = ac > 1 ? av[1] : "data.csv";
  Config c = {5, 250, .02, .01, NULL};
  char *list;
  char *tok;
  if (!load_csv(path, &all)) {
    fprintf(stderr, "Cannot read %s\n", path);
    return 1;
  }
  list = malloc(strlen(ac > 2 ? av[2] : "42") + 1);
  if (!list) {
    fprintf(stderr, "Cannot allocate seed list\n");
    free(all.a);
    return 1;
  }
  strcpy(list, ac > 2 ? av[2] : "42");
  for (tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
    unsigned long long seed = strtoull(tok, NULL, 10);
    unsigned long long base_seed = seed ? seed : 1;
    Rng split_rng = {base_seed};
    Rng hc_rng = {base_seed + 1000003ULL};
    Rng ts_rng = {base_seed + 2000003ULL};
    Rng sa_rng = {base_seed + 3000003ULL};
    Rng ga_rng = {base_seed + 4000003ULL};
    Result results[4];
    const char *names[4] = {"HC", "TS", "SA", "GA"};
    if (!split(&all, &tr, &te, &split_rng) || !make_folds(&tr, c.folds, &c.fold_id)) {
      fprintf(stderr, "Cannot allocate split or folds\n");
      free(tr.a);
      free(te.a);
      free(c.fold_id);
      free(list);
      free(all.a);
      return 1;
    }
    printf("\nSeed %llu (train=%zu, test=%zu)\n", seed, tr.n, te.n);
    for (int i = 0; i < 4; i++) {
      clock_t start = clock();
      results[i] = i == 0   ? hc(&tr, &c, &hc_rng)
                   : i == 1 ? ts(&tr, &c, &ts_rng)
                   : i == 2 ? sa(&tr, &c, &sa_rng)
                            : ga(&tr, &c, &ga_rng);
      if (!show(names[i], results[i], &tr, &te, &c,
                (double)(clock() - start) / CLOCKS_PER_SEC)) {
        fprintf(stderr, "Cannot allocate test-evaluation indices\n");
        free(tr.a);
        free(te.a);
        free(c.fold_id);
        free(list);
        free(all.a);
        return 1;
      }
    }
    free(tr.a);
    free(te.a);
    free(c.fold_id);
    c.fold_id = NULL;
  }
  free(list);
  free(all.a);
  return 0;
}
