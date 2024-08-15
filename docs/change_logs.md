# ChangeLog for Fooocus-API

## [v0.5.0.1]
### Changed
- Fooocus to v2.5.3
- Add enhance image endpoint
- Add generate mask endpoint
- Influenced by Fooocus, the worker.py was reconstructed
- Update docs
- Returned base64 str now include identifier like this `data:image/jpeg;base64,`

### Fixed
- Issue #375
- Issue #378
- Save extension params now also takes effect for base64 str in the returned data

## [v0.4.1.1] - 2024-05-2
### Added
- LoraManager support tar file

### Changed
- Update Fooocus to v2.4.3

### Fixed
- Issue #363

## [v0.4.1.0] - 2024-05-28
### Added
- Add nsfw checker

### Changed
- Fooocus to v2.4.1
- Increase support for config.txt

### Fixed
- Issue #319
- Issue #327
- Issue #332
- Fix JPEG support

## [v0.4.0.6] - 2024-04-24
### Added
- UA in image request

### Changed
- Fooocus to v2.3.1
- Code formatting.

### Fixed
- Issue #302
- Issue #294

## [v0.4.0.5] - 2024-04-16
### Changed
- Sync v1 endpoints.

### Fixed
- meta_scheme error

## [v0.4.0.4] - 2024-04-15
### Added
- Image meta save to image

### Changed
- Code formatting.

## [v0.4.0.3] - 2024-04-12
### Fixed
- Fix opencv-python-headless failed in Docker
- Issue #270

## [v0.4.0.2] - 2024-04-09
### Fixed
- Issue #280
- Issue #222

## [v0.4.0.1] - 2024-04-08
### Added
- Url support for lora in replicate

## [v0.4.0.0] - 2024-04-08
### Changed
- Update docs
- Rewrite Dockerfile
- Rewirte examples
- Full code of Fooocus include
- Remove related code
- Optimize project structure.

## [v0.3.33] - 2024-04-07
### Added
- Support for Lightning model.
- Update default checkpoint for Replicate.

### Fixed
- Issue #244
- Issue #259
- Issue #232

## [v0.3.32] - 2024-03-21
### Added
- Support for Fooocus 2.3.0.

### Changed
- Project structure optimization.

### Fixed
- Removed unnecessary dependencies and optimized code.

## [v0.3.31] - 2024-03-20
### Added
- Save extension
- Secure API future with the use of API keys.
- Add seeds for predict output

### Changed
- Update Docs

### Fixed
- OOM when running a long time
- Some spell error
- Issue with the Fooocus-API version in startup output.

## [v0.3.30] - 2024-01-26
### Added
- Support url for input_image in v2 API.
- Image Prompt Mixing requirements implemented
- Add SQLite database support for history.

### Changed
- Update Docs
- Large queue size support
- Optimized async task response when the queue is full
- Update cog branch
- Optimized cli flages parser.
- Optimized some code formatting.
- Optimized the underlying logic of task execution.
- Default queue history size to 0 for no limit.

### Fixed
- Fix condition. default params broke here and this fixes auto mixing feature.
- Fix error when use `Extreme Speed' with cog.
- Fix typo of 'presistent'
- Image Prompt Mixing requirements implemented
- Some spell error, some translations.
- Fix image prompt must require 'input_image'.
- Implemented support for storing history to the database.

## [v0.3.29] - 2024-01-04
### Added
- Add example using ipynb
- Add error logging
- Add check for aspect_ratios_selection
- Image Prompt Mixing requirements implemented.

### Changed
- Update Docs
- Merge Fooocus to v2.1.860

### Fixed
- Various bugs and issues reported by the community.

## [v0.3.28] - 2024-01-03
### Added
- Add ping endpoint
- Describe interface to get prompts from images.
- Add image_prompt to text2img endpoint
- Add mirror for fooocus

### Changed
- Update Docs
- Added query job history API and webhook_url support for each generation request.
- Change to exit when Fooocus check failed

### Fixed
- Fix #122 query job not found error.

Please note that this ChangeLog is a summary and may not include all changes. For a complete list of changes, please refer to the commit history on the [Fooocus-API GitHub repository](https://github.com/mrhan1993/Fooocus-API).
