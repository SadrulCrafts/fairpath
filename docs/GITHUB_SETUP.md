# Publishing the repository

## Suggested name, description and topics

* **Name:** `fairpath`
  (alternatives: `fair-student-early-warning`, `fairness-aware-student-success`)
* **Description:** *Fairness-aware early warning for student success: leak-free ML pipeline, fairness audit with uncertainty,
  pre/in/post-processing mitigation, a pre-flight check, and an explainable decision-support dashboard.*
* **Topics:** `algorithmic-fairness` `responsible-ai` `fairness` `machine-learning` `educational-data-mining` `learning-analytics`
  `explainable-ai` `shap` `fairlearn` `bias-mitigation` `early-warning-system` `fastapi` `nextjs` `xgboost` `conformal-prediction`
* **Website:** your demo URL once deployed.

## 1. Set your identity on the commits

The history was created with a placeholder e-mail. Point it at the e-mail of your GitHub account so the commits link to your profile:

```bash
cd fairpath
git config user.name  "MD SADRUL ANAM"
git config user.email "<the e-mail on your GitHub account>"
git rebase -r --root --exec "git commit --amend --no-edit --reset-author"
```

## 2. Replace the placeholders

```bash
grep -rln "SadrulCrafts\|<your-demo-url>\|<your-email>\|<your-linkedin>" --exclude-dir=node_modules --exclude-dir=.git .
```
Edit those files (README.md, CITATION.cff, .env.example, docs). On Linux/macOS you can do the username in one go:

```bash
grep -rl "SadrulCrafts" --exclude-dir=node_modules --exclude-dir=.git . | xargs sed -i 's/SadrulCrafts/YOUR_NAME/g'
git commit -am "docs: fill in author links"
```
(macOS: use `sed -i ''`.)

## 3. Create the GitHub repository and push

With the GitHub CLI:

```bash
gh auth login
gh repo create fairpath --public --source=. --remote=origin \
  --description "Fairness-aware early warning for student success" --push
gh repo edit --add-topic algorithmic-fairness,responsible-ai,fairness,machine-learning,educational-data-mining,explainable-ai,shap,fairlearn,bias-mitigation,fastapi,nextjs
```

Without the CLI: create an empty public repository called `fairpath` on github.com (no README, no licence), then:

```bash
git remote add origin https://github.com/SadrulCrafts/fairpath.git
git push -u origin main
```

## 4. Tag the release

```bash
git tag -a v1.0.0 -m "FairPath 1.0.0"
git push origin v1.0.0
gh release create v1.0.0 --title "FairPath 1.0.0" --notes-file docs/RELEASE_NOTES_v1.0.0.md
```

## 5. Polish the repository page

* Pin the repository on your profile.
* In *Settings → General*, tick "Include in the home page: Releases".
* Add the demo URL as the repository website.
* Check that the CI badge turns green after the first push (Actions tab).

## Commit history

The repository already has a readable history in Conventional Commits style, one commit per logical piece of work
(`git log --oneline`). For future work keep the same habit:

```
feat(fairness): add calibration-within-groups metric
fix(api): reject empty CSV uploads with a clear message
docs: explain the pre-flight check in plain language
chore(results): retrain after dependency update
```
