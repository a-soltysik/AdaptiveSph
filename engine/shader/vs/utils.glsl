vec4 eulerToQuat(vec3 eulerAngles) {
    float c1 = cos(eulerAngles.x * 0.5);
    float c2 = cos(eulerAngles.y * 0.5);
    float c3 = cos(eulerAngles.z * 0.5);
    float s1 = sin(eulerAngles.x * 0.5);
    float s2 = sin(eulerAngles.y * 0.5);
    float s3 = sin(eulerAngles.z * 0.5);

    vec4 q;
    q.x = s1 * c2 * c3 - c1 * s2 * s3;// X
    q.y = c1 * s2 * c3 + s1 * c2 * s3;// Y
    q.z = c1 * c2 * s3 - s1 * s2 * c3;// Z
    q.w = c1 * c2 * c3 + s1 * s2 * s3;// W
    return q;
}

mat3 quatToMat3(vec4 q) {
    float x2 = q.x * q.x;
    float y2 = q.y * q.y;
    float z2 = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;

    return mat3(
    vec3(1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)),
    vec3(2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)),
    vec3(2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2))
    );
}
