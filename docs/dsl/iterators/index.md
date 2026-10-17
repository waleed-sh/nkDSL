# Iterators

Iterators define the visit schedule of a term. They determine which labels are in scope and how much work the term performs for one input configuration.

An iterator is a static domain, not a Python loop over sampled configurations.
For each input state, the compiled operator visits the same rows of site
indices. Conditions decide which visits contribute. Each emission on the term
can contribute one candidate connection per visit.

## Choose a domain

For a system with `N` sites, the built-in choices have these visit counts:

| Iterator | Visits per configuration | Labels |
| --- | --- | --- |
| `globally()` | 1 | None |
| `for_each_site("i")` | `N` | One site |
| `for_each_pair("i", "j")` | `N * N` | Two sites, including equal indices |
| `for_each_distinct_pair("i", "j")` | `N * (N - 1)` | Two different sites |
| `for_each(labels, over=rows)` | `len(rows)` | One label per column |

For graph-local operators, prefer explicit edge or plaquette rows over a dense
pair sweep with a filter. The domain then records the geometry directly and
avoids generating irrelevant visits. Dense pair iterators are useful for
all-to-all interactions.

Every iterator call opens a new term and seals the previous one. Labels belong
to that term, so a condition or amplitude can only use labels bound by its
iterator. See [chaining](../chaining.md) for how this works in a longer
definition. The pages below explain ordering, repeated rows, and typical uses.

- [``globally()``](globally.md)
- [``for_each_site(label)``](for_each_site.md)
- [``for_each_pair(label_a, label_b)``](for_each_pair.md)
- [``for_each_distinct_pair(label_a, label_b)``](for_each_distinct_pair.md)
- [Static K-body iterators](for_each_kbody.md)
