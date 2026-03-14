# KernelEvalPlus Version Management

This document describes the versioning and release process for KernelEvalPlus.

## Version Format

KernelEvalPlus follows [Semantic Versioning 2.0.0](https://semver.org/):

```
v<MAJOR>.<MINOR>.<PATCH>
```

Example: `v0.1.0`, `v1.0.0`, `v1.2.3`

### Version Components

- **MAJOR**: Incompatible API changes or major architecture changes
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible

## Release Process

### 1. Prepare Release

Update version-related files:
- `README.md` - Update version badges
- `CHANGELOG.md` - Document changes
- Update benchmark data if needed

### 2. Create Version Tag

```bash
# Format: v<MAJOR>.<MINOR>.<PATCH>
git tag -a v0.1.0 -m "Release v0.1.0: Initial public release"

# View tag
git show v0.1.0

# Push tag to GitLab
git push origin v0.1.0
```

### 3. Automated Deployment

Pushing a version tag automatically triggers:
1. GitLab CI/CD pipeline
2. Build benchmark static site
3. Deploy to GitLab Pages
4. Create release artifacts

### 4. GitLab Release (Optional)

Create a formal release in GitLab:

1. Go to **Deployments** → **Releases**
2. Click **New Release**
3. Select the tag (e.g., `v0.1.0`)
4. Add release notes:
   - Overview of changes
   - New features
   - Bug fixes
   - Breaking changes (if any)
5. Attach artifacts (optional):
   - Benchmark data CSV
   - Documentation PDF
   - Build artifacts

## Version History Example

### v0.3.0 (2026-02-13)
**Features:**
- Add GitLab Pages deployment
- CSV-based benchmark visualization
- Hardware classification (Laptop/Desktop/Server)
- Static site generation

**Improvements:**
- Enhanced baseline comparison
- Better WebUI performance

**Bug Fixes:**
- Fix hardware detection logic
- Correct baseline mapping

### v0.2.0 (2026-02-01)
**Features:**
- Initial benchmark framework
- GGML baseline integration
- Multi-GPU support

### v0.1.0 (2026-01-15)
**Features:**
- Initial public release
- Basic CUDA kernel testing
- Performance metrics

## Tag Management

### List All Tags

```bash
# List all tags
git tag

# List tags with messages
git tag -n

# List tags matching pattern
git tag -l "v0.1.*"
```

### Delete Tag

```bash
# Delete local tag
git tag -d v0.1.0

# Delete remote tag
git push origin --delete v0.1.0
```

### Move Tag (Not Recommended)

```bash
# Force update tag to current commit
git tag -f v0.1.0

# Force push to remote
git push origin v0.1.0 --force
```

**Note:** Avoid moving tags after they're published, as this can break deployments.

## Continuous Deployment

### Main Branch
- Commits to `main` → Deploy to production Pages
- Automatic deployment on every push
- No manual intervention needed

### Version Tags
- Create tag → Trigger deployment
- Permanent deployment record
- Accessible via GitLab Releases

### Merge Requests
- Create MR → Deploy to preview environment
- Preview URL in MR comments
- Auto-cleanup after 1 week

## Best Practices

1. **Always create annotated tags** (`-a` flag) with descriptive messages
2. **Follow semantic versioning** strictly
3. **Update CHANGELOG.md** before creating tags
4. **Test thoroughly** before tagging
5. **Never delete or move published tags**
6. **Use descriptive tag messages** explaining the release

## Example Tag Creation Workflow

```bash
# 1. Ensure you're on main branch
git checkout main
git pull origin main

# 2. Make and commit changes
git add .
git commit -m "feat: Add new benchmark feature"

# 3. Create annotated tag
git tag -a v0.3.1 -m "$(cat <<EOF
Release v0.3.1: Enhanced benchmark visualization

Features:
- Add interactive charts
- Improve baseline comparison
- Add hardware classification

Bug Fixes:
- Fix GPU detection on laptop
- Correct GFLOPS calculation
EOF
)"

# 4. Push commits and tags
git push origin main
git push origin v0.3.1

# 5. Verify deployment
# Check GitLab CI/CD pipeline status
# Visit GitLab Pages URL
```

## References

- [Semantic Versioning](https://semver.org/)
- [GitLab Releases](https://docs.gitlab.com/ee/user/project/releases/)
- [Git Tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging)
