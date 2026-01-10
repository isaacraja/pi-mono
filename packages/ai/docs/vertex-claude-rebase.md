# Vertex Claude minimal branch rebase

This branch keeps Vertex Claude support isolated to reduce rebase conflicts.

## Recommended flow

```bash
git fetch upstream
git switch maint/vertex-claude-minimal
git rebase upstream/main
```

## Conflict hotspots

If conflicts occur, they should be limited to these files:

- packages/ai/src/providers/google-vertex.ts
- packages/ai/src/providers/google-vertex-claude.ts
- packages/ai/scripts/generate-models.ts
- packages/ai/src/models.generated.ts

## Rerere

Enable conflict reuse once:

```bash
git config --global rerere.enabled true
```

Then use the same rebase flow. After resolving conflicts once, Git will reuse the resolutions.

## After rebase

```bash
npm run check
git push --force-with-lease
```
