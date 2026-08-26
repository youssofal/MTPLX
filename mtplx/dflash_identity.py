"""Construction-time identity gate for the pinned DFlash dependency."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
import json


DFLASH_DISTRIBUTION = "dflash-mlx"
PINNED_DFLASH_VCS = "git"
PINNED_DFLASH_URL = "https://github.com/davidtai/dflash-mlx.git"
PINNED_DFLASH_COMMIT = "54644e991039110f30140006c892c57734b9311e"


@dataclass(frozen=True, slots=True)
class InstalledDFlashIdentity:
    """Frozen PEP 610 receipt for the DFlash source used by this process."""

    vcs: str
    url: str
    commit_id: str
    requested_revision: str


PINNED_DFLASH_IDENTITY = InstalledDFlashIdentity(
    vcs=PINNED_DFLASH_VCS,
    url=PINNED_DFLASH_URL,
    commit_id=PINNED_DFLASH_COMMIT,
    requested_revision=PINNED_DFLASH_COMMIT,
)


def assert_pinned_dflash_identity(
    identity: InstalledDFlashIdentity,
) -> InstalledDFlashIdentity:
    """Reject a receipt that is not the sealed Mia DFlash source."""

    if identity != PINNED_DFLASH_IDENTITY:
        raise RuntimeError(
            "installed DFlash source does not match the sealed Mia runtime: "
            f"observed={identity!r}, expected={PINNED_DFLASH_IDENTITY!r}"
        )
    return identity


def require_pinned_dflash_install() -> InstalledDFlashIdentity:
    """Read and validate the installed DFlash PEP 610 receipt."""

    try:
        package = distribution(DFLASH_DISTRIBUTION)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"{DFLASH_DISTRIBUTION} is not installed; expected "
            f"{PINNED_DFLASH_URL}@{PINNED_DFLASH_COMMIT}"
        ) from exc
    direct_url_text = package.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(
            f"{DFLASH_DISTRIBUTION} has no PEP 610 direct_url.json; expected "
            f"{PINNED_DFLASH_URL}@{PINNED_DFLASH_COMMIT}"
        )
    try:
        direct_url = json.loads(direct_url_text)
        vcs_info = direct_url["vcs_info"]
        identity = InstalledDFlashIdentity(
            vcs=str(vcs_info["vcs"]),
            url=str(direct_url["url"]),
            commit_id=str(vcs_info["commit_id"]),
            requested_revision=str(vcs_info["requested_revision"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{DFLASH_DISTRIBUTION} has an invalid PEP 610 VCS receipt"
        ) from exc
    return assert_pinned_dflash_identity(identity)
