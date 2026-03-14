# Changelog

All notable changes to KernelEvalPlus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitLab Pages deployment with automated CI/CD pipeline
- Static benchmark site with interactive visualization
- CSV-based benchmark viewer (`app_csv.py`)
- Hardware classification system (Laptop/Desktop/Server)
- Comprehensive documentation:
  - `HARDWARE_CLASSIFICATION.md` - GPU categorization guide
  - `VERSION_MANAGEMENT.md` - Release and tagging guide
  - `GITLAB_DEPLOY.md` - Deployment instructions

### Changed
- Improved `.gitlab-ci.yml` with build/deploy stages and tags
- Enhanced hardware detection to properly classify RTX 4070/5070 as laptop GPUs
- Updated baseline comparison logic for better accuracy
- Cleaned up redundant WebUI files

### Removed
- Removed `app_v2.py` (duplicate/broken version)
- Removed `app.py.bak` (backup file)

### Fixed
- Hardware type detection now correctly identifies laptop GPUs
- Baseline hardware mapping improved for different GPU categories

## [0.2.0] - 2026-02-XX

### Added
- Initial benchmark framework
- GGML baseline integration
- Multi-GPU support
- WebUI for result visualization

## [0.1.0] - 2026-01-XX

### Added
- Initial public release
- Basic CUDA kernel testing framework
- Performance metrics collection
- Reference implementation comparison

[Unreleased]: https://gitlab.com/username/kernelevalplus/compare/v0.2.0...HEAD
[0.2.0]: https://gitlab.com/username/kernelevalplus/compare/v0.1.0...v0.2.0
[0.1.0]: https://gitlab.com/username/kernelevalplus/releases/tag/v0.1.0
