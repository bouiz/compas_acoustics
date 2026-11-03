"""Edge diffraction by the Biot-Tolstoy-Medwin-Svensson (BTMS) model.

Svensson, Fred and Vanderkooy (1999), "An analytic secondary source model of
edge diffraction impulse responses", J. Acoust. Soc. Am. 106(5), 2331-2344.
Time-domain kernels in the edge-cylindrical coordinates of EDtoolbox: z along
the edge, theta from the reference face into the open region, closed wedge
angle 2 pi - theta_w, free field 1/r.

wedge1st_ir follows EDwedge1st_ir, wedge2nd_ir EDwedge2nd and _coordtrans1
EDcoordtrans1 of Svensson's Edge Diffraction Toolbox v0.504:

    Copyright (c) 2018, Peter Svensson All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDEN- TAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUD- ING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTER- RUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

betaoverml and _quadstep are written from the paper and the adaptive Simpson
rule, not from the toolbox.
"""

from __future__ import annotations

import numpy as np


def betaoverml(z, rs, rr, zs, zr, nu, sin_nu_phi, cos_nu_phi):
    """beta(z) / (m l): the four-term edge-source directivity over the two
    distances, Svensson et al. (1999); nu = pi / theta_w, phi = pi +- theta_S
    +- theta_R."""
    z = np.asarray(z, dtype=float)
    m = np.sqrt((z - zs) ** 2 + rs**2)
    l = np.sqrt((z - zr) ** 2 + rr**2)
    y = (m * l + (z - zs) * (z - zr)) / (rs * rr)
    cosh_nu_eta = np.cosh(nu * np.arccosh(np.maximum(y, 1.0)))
    # a term on its zone boundary is 0/0 where the path reaches the apex (z = 0 for a
    # segment starting there); the apex region drops it, so does the integrand
    den = [cosh_nu_eta - cphi for cphi in cos_nu_phi]
    beta = sum(np.where(d == 0, 0.0, s / np.where(d == 0, 1.0, d)) for s, d in zip(sin_nu_phi, den))
    return beta / (m * l)


def _quadstep(a, b, fa, fc, fb, tol, rs, rr, zs, zr, nu, sin_nu_phi, cos_nu_phi):
    """Adaptive Simpson on [a, b] for every interval at once: three-point and
    five-point Simpson, one Richardson step, intervals whose estimates differ
    by more than tol are halved."""
    c = (a + b) / 2.0
    fd, fe = (betaoverml(x, rs, rr, zs, zr, nu, sin_nu_phi, cos_nu_phi)
              for x in ((a + c) / 2.0, (c + b) / 2.0))
    h = (b - a) / 6.0
    q1 = h * (fa + 4.0 * fc + fb)
    q2 = h / 2.0 * (fa + 4.0 * fd + 2.0 * fc + 4.0 * fe + fb)
    q = q2 + (q2 - q1) / 15.0
    open_ = np.flatnonzero(np.abs(q2 - q) > tol)
    if open_.size:
        q[open_] = (_quadstep(a[open_], c[open_], fa[open_], fd[open_], fc[open_], tol,
                              rs, rr, zs, zr, nu, sin_nu_phi, cos_nu_phi)
                    + _quadstep(c[open_], b[open_], fc[open_], fe[open_], fb[open_], tol,
                                rs, rr, zs, zr, nu, sin_nu_phi, cos_nu_phi))
    return q


def _round_half_away(x):
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)


def _slot_integrals(zstarts, zends, tol, rs, rr, zs, zr, nu, s, c):
    h = 0.13579 * (zends - zstarts)
    x = np.stack([zstarts, zstarts + h, zstarts + 2 * h, (zstarts + zends) / 2,
                  zends - 2 * h, zends - h, zends], axis=1)
    y = betaoverml(x, rs, rr, zs, zr, nu, s, c)
    args = (tol, rs, rr, zs, zr, nu, s, c)
    q = np.stack([_quadstep(x[:, 0], x[:, 2], y[:, 0], y[:, 1], y[:, 2], *args),
                  _quadstep(x[:, 2], x[:, 4], y[:, 2], y[:, 3], y[:, 4], *args),
                  _quadstep(x[:, 4], x[:, 6], y[:, 4], y[:, 5], y[:, 6], *args)], axis=1)
    return q.sum(axis=1)


def _edge_z_of_path(x, rs, rr, zs, zr):
    x2 = x**2
    K = rs * rs + zs * zs - (rr * rr + zr * zr) - x2
    diffz = zs - zr
    denom = diffz**2 - x2
    a = (2 * x2 * zr - K * diffz) / denom
    b = (K**2 / 4 - x2 * (rr * rr + zr * zr)) / denom
    # x = R0 gives zero up to rounding, and a segment ending on the apex reaches it
    return -a / 2 + np.sqrt(np.maximum(a**2 / 4 - b, 0.0))


def wedge1st_ir(fs, c, closed_angle, rs, thetas, zs, rr, thetar, zr, zw, r_irstart=0.0):
    """First-order diffraction impulse response of a finite rigid wedge for a
    point source (free field 1/r), EDwedge1st_ir.m, method 'New'.

    Returns (ir, initdelay, singular, zfirst); ir[0] is the sample at
    r_irstart / c, singular flags the beta terms on a zone boundary, zfirst is
    None for an edge of zero length."""
    zw = np.asarray(zw, dtype=float)
    if zw[1] - zw[0] == 0:
        return np.zeros(2), 0.0, np.zeros(4, dtype=bool), None

    eps = np.finfo(float).eps
    ny = np.pi / (2 * np.pi - closed_angle)
    initdelay = r_irstart / c
    sampconst = c / fs
    tol = 1e-11
    zrelmax = 0.1

    za = (rs * zr + rr * zs) / (rs + rr)
    zfirst = za
    zs = zs - za
    zr = zr - za
    zw = zw - za

    R0 = np.sqrt((rs + rr) ** 2 + (zr - zs) ** 2)

    fivec = np.pi + thetas * np.array([1, 1, -1, -1]) + thetar * np.array([1, -1, 1, -1])
    absnyfivec = np.abs(ny * fivec)
    sinnyfivec = np.sin(ny * fivec)
    cosnyfivec = np.cos(ny * fivec)
    integ = (rs, rr, zs, zr, ny, sinnyfivec, cosnyfivec)

    perpcase = zs == zr
    symmcase = rs == rr
    skewcase = not perpcase and not symmcase
    apexincluded = np.sign(zw[0] * zw[1]) != 1

    Rendnear = np.sqrt(rs**2 + (zw[0] - zs) ** 2) + np.sqrt(rr**2 + (zw[0] - zr) ** 2)
    Rendfar = np.sqrt(rs**2 + (zw[1] - zs) ** 2) + np.sqrt(rr**2 + (zw[1] - zr) ** 2)
    if Rendnear > Rendfar:
        Rendnear, Rendfar = Rendfar, Rendnear
    tailincluded = not abs(Rendfar - Rendnear) < eps * 1e6

    arrivalsampnumb = int(_round_half_away((R0 - r_irstart) / sampconst))
    if apexincluded and arrivalsampnumb < 0:
        raise ValueError(
            f"r_irstart ({r_irstart:.6g} m) is later than the apex arrival ({R0:.6g} m) by more than half a sample")
    endsampnumb_apex = int(_round_half_away((Rendnear - r_irstart) / sampconst))
    if tailincluded:
        endsampnumb_tail = int(_round_half_away((Rendfar - r_irstart) / sampconst))
        ir = np.zeros(endsampnumb_tail + 1)
    else:
        ir = np.zeros(endsampnumb_apex + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        if apexincluded:
            x = sampconst * (arrivalsampnumb + np.arange(endsampnumb_apex - arrivalsampnumb + 1) + 0.5) + r_irstart
            if x[-1] > Rendnear:
                x[-1] = Rendnear
                if x.size > 1 and x[-1] < x[-2]:
                    raise ValueError("apex integration intervals are not increasing")
            zrange_apex = _edge_z_of_path(x, rs, rr, zs, zr)
            zrangespecial = zrelmax * min(rs, rr)
            if zrangespecial > zrange_apex[0]:
                zrangespecial = abs(zrange_apex[0])
                splitinteg = False
            else:
                splitinteg = True

        if tailincluded:
            x = sampconst * (endsampnumb_apex + np.arange(endsampnumb_tail - endsampnumb_apex + 2) - 0.5) + r_irstart
            if x[-1] > Rendfar:
                x[-1] = Rendfar
                if x.size > 1 and x[-1] < x[-2]:
                    raise ValueError("tail integration intervals are not increasing")
            if x[0] < Rendnear:
                x[0] = Rendnear
                if x.size > 1 and x[1] < x[0]:
                    raise ValueError("tail integration intervals are not increasing")
            zrange_tail = _edge_z_of_path(x, rs, rr, zs, zr)

        singularterm = np.zeros(4, dtype=bool)

        if apexincluded:
            rho = rr / rs
            sinpsi = (rs + rr) / R0
            cospsi = (zr - zs) / R0
            tempfact = (1 + rho) ** 2 * sinpsi**2 - 2 * rho

            useserialexp1 = absnyfivec < 0.01
            useserialexp2 = np.abs(absnyfivec - 2 * np.pi) < 0.01
            useserialexp = useserialexp1 | useserialexp2
            singularterm = (absnyfivec < 10 * eps) | (np.abs(absnyfivec - 2 * np.pi) < 10 * eps)

            # np.where, not a 0/1 factor: the unused series branch takes sqrt of a negative number
            sqrtB1vec = (np.sqrt(2 * (1 - cosnyfivec)) * R0 * rho / (1 + rho) ** 2 / ny * ~useserialexp
                         + np.where(useserialexp1, fivec * np.sqrt(1 - (ny * fivec) ** 2 / 12), 0.0)
                         * R0 * rho / (1 + rho) ** 2
                         + np.where(useserialexp2, (fivec - 2 * np.pi / ny)
                                    * np.sqrt(1 - (ny * fivec - 2 * np.pi) ** 2 / 12), 0.0)
                         * R0 * rho / (1 + rho) ** 2)

            if not skewcase:
                # tempfact < 0 (rs == rr, long path) makes sqrtB3 imaginary, as in the toolbox
                sqrtB3 = np.sqrt(2) * R0 * rho / (1 + rho) / np.sqrt(complex(tempfact))
                usespecialcase = np.abs(sqrtB3 - sqrtB1vec) < 1e-14

                fifactvec = ((1 - cosnyfivec) / ny**2 * ~useserialexp
                             + fivec**2 / 2 * (1 - (ny * fivec) ** 2 / 12) * useserialexp1
                             + (fivec - 2 * np.pi / ny) ** 2 / 2 * (1 - (ny * fivec - 2 * np.pi) ** 2 / 12)
                             * useserialexp2)

                temp1vec = sinnyfivec / ((1 + rho) ** 2 - tempfact * fifactvec + eps * 10)
                temp1_2vec = ((sinnyfivec + 10 * eps) / ((1 + rho) ** 2 - tempfact * fifactvec)
                              / (sqrtB1vec + 10 * eps) * np.arctan(zrangespecial / (sqrtB1vec + eps)))
                temp3vec = -1 / sqrtB3 * np.arctan(zrangespecial / sqrtB3)

                approxintvalvec = 2 / ny**2 * rho * (temp1_2vec + temp1vec * temp3vec)
                approxintvalvec = approxintvalvec * (sinnyfivec != 0) * ~usespecialcase

                if np.any(usespecialcase):
                    specialcasevalue = 1 / (2 * sqrtB3 * sqrtB3) * (
                        zrangespecial / (zrangespecial**2 + sqrtB3 * sqrtB3)
                        + 1 / sqrtB3 * np.arctan(zrangespecial / (sqrtB3 + eps)))
                    approxintvalvec = approxintvalvec + (
                        4 * R0 * R0 * rho**3 * sinnyfivec / ny / ny / (1 + rho) ** 4 / tempfact
                        * specialcasevalue * usespecialcase)
            else:
                B3 = 2 * R0 * R0 * rho * rho / (1 + rho) / (1 + rho) / tempfact
                B1vec = sqrtB1vec**2
                B2 = -2 * R0 * (1 - rho) * rho * cospsi / (1 + rho) / tempfact
                E1vec = 4 * R0**2 * rho**3 * sinnyfivec / ny**2 / (1 + rho) ** 4 / tempfact
                multfact = -E1vec * B2 / (B1vec * B2**2 + (B1vec - B3) ** 2)

                P1 = 0.5 * np.log(abs(B3) * np.abs(zrangespecial**2 + B1vec) / np.abs(B1vec)
                                  / np.abs(zrangespecial**2 + B2 * zrangespecial + B3))
                P2 = (B1vec - B3) / (sqrtB1vec + 10 * eps) / B2 * np.arctan(zrangespecial / (sqrtB1vec + 10 * eps))
                q = 4 * B3 - B2**2

                if q > 0:
                    sqrtq = np.sqrt(q)
                    F = 2 / sqrtq * (np.arctan((2 * zrangespecial + B2) / sqrtq) - np.arctan(B2 / sqrtq))
                elif q < 0:
                    sqrtminq = np.sqrt(-q)
                    F = 1 / sqrtminq * np.log(abs(2 * zrangespecial + B2 - sqrtminq) * abs(B2 + sqrtminq)
                                              / abs(2 * zrangespecial + B2 + sqrtminq) / abs(B2 - sqrtminq))
                else:
                    F = 4 * zrangespecial / B2 / (2 * zrangespecial + B2)
                P3 = (2 * (B3 - B1vec) - B2**2) / 2 / B2 * F

                approxintvalvec = multfact * (P1 + P2 + P3)

            # np.where, not the toolbox's factor 1 - singularterm: a singular term can be NaN, and NaN * 0 = NaN
            approxintvalvec = np.sum(np.where(singularterm, 0.0, approxintvalvec))

            ir[arrivalsampnumb] += approxintvalvec.real

            if splitinteg:
                ir[arrivalsampnumb] += _slot_integrals(
                    np.array([zrangespecial]), zrange_apex[:1], tol, *integ)[0]

            qsum = _slot_integrals(zrange_apex[:-1], zrange_apex[1:], tol, *integ)
            # the toolbox's ir(arrivalsampnumb + [2:length(zrange_apex)]), 1-based
            ir[arrivalsampnumb + 1:arrivalsampnumb + zrange_apex.size] += qsum
            ir = ir * (-ny / 2 / np.pi)
        else:
            Rend1 = np.sqrt(rs**2 + (zw[0] - zs) ** 2) + np.sqrt(rr**2 + (zw[0] - zr) ** 2)
            Rend2 = np.sqrt(rs**2 + (zw[1] - zs) ** 2) + np.sqrt(rr**2 + (zw[1] - zr) ** 2)
            zfirst = 0.0 if Rend1 < Rend2 else zw[1] - zw[0]

        if tailincluded:
            qsum = _slot_integrals(zrange_tail[:-1], zrange_tail[1:], tol, *integ)
            # the toolbox's ir(endsampnumb_apex + [1:length(zrange_tail)-1]), 1-based
            ir[endsampnumb_apex:endsampnumb_apex + zrange_tail.size - 1] += qsum * (-ny / 4 / np.pi)

    return ir, initdelay, singularterm, float(zfirst)


def _df(nu, b, angle):
    return np.sin(nu * angle) / (b - np.cos(nu * angle))


def _coordtrans1(points, edge_ends, ref_normal):
    """Cylindrical coordinates (r, theta, z) of points about an edge, EDcoordtrans1.m
    (several points): origin at edge_ends[0], z along the edge, theta = 0 on the
    reference plane with normal ref_normal."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    edge_ends = np.asarray(edge_ends, dtype=float)
    n = np.asarray(ref_normal, dtype=float)
    xknown = edge_ends[1] - edge_ends[0]
    xknown = xknown / np.sqrt(np.sum(xknown**2))
    cross = np.cross(n, xknown)
    a = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]) @ np.linalg.inv(
        np.stack([xknown, cross, n], axis=1))
    xyz = (a @ (points - edge_ends[0]).T).T
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    z = xyz[:, 2]
    theta = np.zeros(points.shape[0])
    iv = r > 0
    x, y, ri = xyz[iv, 0], xyz[iv, 1], r[iv]
    t = np.real(np.arccos(x / ri) * (y != 0))
    t = t + np.pi * ((y == 0) & (x < 0))
    theta[iv] = np.where(y >= 0, t, 2 * np.pi - t)
    return r, theta, z


def element_grid(lengths, c, fs):
    """Edge-element midpoints of an edge pair as fractions of the two edge
    lengths, rows in the order of EDcreindexmatrix (edge 2 fastest), and the
    element lengths dz (EDmakeHODirs.m, order 2, element size c / fs)."""
    lengths = np.asarray(lengths, dtype=float)
    ndiv = np.ceil(lengths / (c / fs)).astype(int)
    dz = lengths / ndiv
    idx = np.stack(np.meshgrid(np.arange(1, ndiv[0] + 1), np.arange(1, ndiv[1] + 1), indexing="ij"),
                   axis=-1).reshape(-1, 2)
    return (idx - 0.5) / ndiv, dz


def wedge2nd_ir(cyl_s, cyl_r, cyl_e2_r1, cyl_e1_r2, nu, edge_limits, dz, steps, path_along_plane,
                r_irstart, c, fs):
    """Edge-to-edge (second-order) diffraction impulse response of a pair of
    rigid wedges, EDwedge2nd.m with method 'n' and swapbigmatrix = 0.

    cyl_s, cyl_r are (r, theta, z) of source and receiver about edges 1 and 2;
    cyl_e2_r1 are the two ends of edge 2 about edge 1, cyl_e1_r2 the ends of
    edge 1 about edge 2; nu = (pi / theta_w1, pi / theta_w2); edge_limits rows
    are [start, end] along each edge; steps and dz are the element_grid
    midpoints and element lengths; path_along_plane selects the form for two
    edges bounding a common plane.
    ir[0] is the sample at r_irstart / c."""
    nu1, nu2 = nu
    nu_closed2 = np.pi / nu2
    multfac = 1.0 / (float(bool(path_along_plane)) + 1.0)
    rs, thetas, zs = cyl_s
    rr, thetar, zr = cyl_r
    r_e1_r1, theta_e1_r1, z_e1_r1 = cyl_e2_r1[0]
    r_e2_r1, theta_e2_r1, z_e2_r1 = cyl_e2_r1[1]
    r_e1_r2, theta_e1_r2, z_e1_r2 = cyl_e1_r2[0]
    r_e2_r2, theta_e2_r2, z_e2_r2 = cyl_e1_r2[1]

    closwedang = 2 * np.pi - nu_closed2
    in_plane = ((abs(theta_e1_r1) < 1e-10 or abs(theta_e1_r1 - (2 * np.pi - closwedang)) < 1e-10)
                and abs(theta_e1_r1 - theta_e2_r1) < 1e-10)

    s1, s2 = steps[:, 0], steps[:, 1]
    b4 = edge_limits[0, 0] + (edge_limits[0, 1] - edge_limits[0, 0]) * s1
    zedge2_re1 = z_e1_r1 + s2 * (z_e2_r1 - z_e1_r1)
    b5 = edge_limits[1, 0] + (edge_limits[1, 1] - edge_limits[1, 0]) * s2
    zedge1_re2 = z_e1_r2 + s1 * (z_e2_r2 - z_e1_r2)

    s2e = np.sqrt(rs**2 + (b4 - zs) ** 2)
    e2r = np.sqrt(rr**2 + (b5 - zr) ** 2)

    if not in_plane:
        x1, y1 = r_e1_r1 * np.cos(theta_e1_r1), r_e1_r1 * np.sin(theta_e1_r1)
        x2, y2 = r_e2_r1 * np.cos(theta_e2_r1), r_e2_r1 * np.sin(theta_e2_r1)
        pts = np.stack([x1 + (x2 - x1) * s2, y1 + (y2 - y1) * s2, z_e1_r1 + (z_e2_r1 - z_e1_r1) * s2], axis=1)
        redge2_re1, thetaedge2_re1, _ = _coordtrans1(
            pts, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, edge_limits[0, 1]]]), np.array([0.0, 1.0, 0.0]))

        x1, y1 = r_e1_r2 * np.cos(theta_e1_r2), r_e1_r2 * np.sin(theta_e1_r2)
        x2, y2 = r_e2_r2 * np.cos(theta_e2_r2), r_e2_r2 * np.sin(theta_e2_r2)
        pts = np.stack([x1 + (x2 - x1) * s1, y1 + (y2 - y1) * s1, z_e1_r2 + (z_e2_r2 - z_e1_r2) * s1], axis=1)
        redge1_re2, thetaedge1_re2, _ = _coordtrans1(
            pts, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, edge_limits[1, 1]]]), np.array([0.0, 1.0, 0.0]))
    else:
        redge2_re1 = r_e1_r1 + s2 * (r_e2_r1 - r_e1_r1)
        redge1_re2 = r_e1_r2 + s1 * (r_e2_r2 - r_e1_r2)
        thetaedge1_re2 = theta_e1_r2
        thetaedge2_re1 = theta_e1_r1

    e2e_2 = np.sqrt((b5 - zedge1_re2) ** 2 + redge1_re2**2)
    b2 = ((b5 - zedge1_re2) * (b5 - zr) + e2e_2 * e2r) / redge1_re2 / rr
    disc = b2**2 - 1
    # the toolbox turns the whole array complex if rounding puts any disc below zero
    root = np.sqrt(disc) if np.all(disc >= 0) else np.sqrt(disc.astype(complex))
    b2 = (root + b2) ** nu2
    b2 = np.real(b2 + 1 / b2) / 2

    e2e = np.sqrt((b4 - zedge2_re1) ** 2 + redge2_re1**2)
    b1 = ((b4 - zs) * (b4 - zedge2_re1) + s2e * e2e) / rs / redge2_re1
    b1 = np.sqrt(np.maximum(b1**2 - 1, 0.0)) + b1
    b1 = b1**nu1
    b1 = (b1 + 1 / b1) / 2

    def four(nuv, bq, a, t):
        return sum(_df(nuv, bq, np.pi + sa * a + st * t) for sa in (1, -1) for st in (1, -1))

    def two(nuv, bq, a, t):
        return sum(_df(nuv, bq, np.pi + a + st * t) for st in (1, -1))

    if not path_along_plane:
        b3 = (multfac * nu1 * nu2 / 16 / np.pi**2 * dz[0] * dz[1]
              * four(nu1, b1, thetas, thetaedge2_re1) / s2e / e2e)
        b3 = b3 * four(nu2, b2, thetaedge1_re2, thetar) / e2r
    else:
        if thetas == 0:
            b3 = (multfac * nu1 * nu2 / 2 / np.pi**2 * dz[0] * dz[1]
                  * _df(nu1, b1, np.pi + thetaedge2_re1) / s2e / e2e)
        else:
            b3 = (multfac * nu1 * nu2 / 4 / np.pi**2 * dz[0] * dz[1]
                  * (_df(nu1, b1, np.pi + thetas + thetaedge2_re1) + _df(nu1, b1, np.pi - thetas + thetaedge2_re1))
                  / s2e / e2e)
        b3 = b3 * two(nu2, b2, thetaedge1_re2, thetar) / e2r

    slots = (s2e + e2e + e2r - r_irstart) / (c / fs) + 1
    b1i = np.floor(slots).astype(int)
    frac = slots - b1i
    first, last = b1i.min(), b1i.max()
    ir = np.zeros(last + 1)
    ir[first - 1:last] = np.bincount(b1i - first, weights=b3 * (1 - frac))
    ir[first:last + 1] += np.bincount(b1i - first, weights=b3 * frac)
    return ir
