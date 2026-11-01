# Demo video script (about 2 minutes 40 seconds)

Record at 1920x1080, browser zoom 110%, dark mode off for the first half and on for the fairness page (it shows the theme
toggle without having to talk about it). Speak slowly; leave a beat after each number. Read the script a few times first,
then talk it rather than reading it word for word.

| Time | On screen | Voice-over |
|---|---|---|
| 0:00–0:15 | Overview page, slow scroll | "Hi, I'm Sadrul. My undergraduate thesis asked a simple question: when can we actually make a student-success model fairer, and when are we just measuring noise? This is FairPath, the open-source system I built from that thesis." |
| 0:15–0:35 | Hover the KPI cards and the findings list | "It runs on two public datasets from Portugal: 3,630 university students and 395 secondary-school students. Everything you see here comes from a real run, and the full pipeline is reproducible with one command." |
| 0:35–1:05 | Predict page → click "Higher-risk example" → result appears | "Here's the advisor's view. I'll load a synthetic student. The model estimates the risk of dropping out, tells you how confident it is, and explains which factors pushed the estimate up or down. Notice the model never sees gender. That's a deliberate design choice, which I'll come back to." |
| 1:05–1:25 | Scroll to "Path to lower risk" | "Below that are support options. These are counterfactuals, but restricted to things an institution can actually act on, like tuition support or a scholarship. It will never suggest changing someone's age or gender." |
| 1:25–1:40 | What-if page, move a slider, watch the gauge | "The what-if page lets an advisor explore those changes interactively." |
| 1:40–2:15 | Toggle dark mode → Fairness page → Pareto chart, then the pre-flight panel | "Now the part the thesis is about. On the dropout data, every mitigation method reduced the gap in error rates between men and women. The deployed model halves the equalized-odds gap on held-out students, at almost no cost in accuracy. Switch to the small dataset" — *switch dataset* — "and the pre-flight check says: insufficient data. On 395 students, a fairness change can't be measured reliably, and the methods that look perfectly fair are flagging everyone or no one." |
| 2:15–2:30 | Models page, then Methodology page | "There's a full model comparison with proper statistics, and a methodology page that explains every choice, including what went wrong in my original code and how I fixed it." |
| 2:30–2:40 | Back to Overview, GitHub link visible | "The code, model card and datasheet are on GitHub. Thanks for watching." |

## Recording checklist

* Close other tabs and notifications. Use the deployed URL, not localhost.
* Wake the API a minute before recording (open `/api/health`), so the first prediction isn't a cold start.
* Record the screen and voice separately if you can; it is easier to fix a sentence than to re-record a scroll.
* Add captions (YouTube auto-captions, then correct them). Many reviewers watch muted.
* Upload as unlisted to YouTube and put the link in the README and your CV.
