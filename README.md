# thedu

Can an LLM that rewrites search queries over plain keyword search (BM25) match
embedding search, and at what cost? This repo is the experiment behind an
[INTERNALS.md](https://internals.laxmena.com) post.

**Status:** research code. Every number in the post can be rebuilt from here, but
this is not a maintained library.

## Quick start

```
uv sync
uv run thedu --compare "do vitamins prevent colds?"   # search, with embeddings side by side
uv run thedu check   # fast tests
uv run thedu dev     # run the current config on dev questions
uv run thedu all     # rebuild every number in the post
```

## More

- [AGENTS.md](AGENTS.md): how the code is organized and the rules for changing it
- [JOURNAL.md](JOURNAL.md): every decision, attempt, dead end and finding, in order

## How do you say "thedu"?

**thedu** (தேடு) is Tamil for "search!", the command form of the verb.

Say it **THAY-doo**, stress on the first part.

| Part | Sounds like | Tip |
| --- | --- | --- |
| **the** (தே) | "thay", as in *they* | The *th* is not the *th* in *think*. It's a soft *t* with your tongue touching the back of your top teeth, like the *t* in Spanish *tú*. |
| **du** (டு) | "doo", but short | Curl your tongue tip back a little before the *d*. The ending is quick and relaxed, closer to "duh-oo" than a long "dooo". |

Close enough is perfect. If it comes out as "tay-doo", every Tamil speaker will
still know what you mean.
