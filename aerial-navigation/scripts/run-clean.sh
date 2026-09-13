#!/usr/bin/env bash
# Run a command with the VS Code snap's confinement environment removed.
# The snap exports GTK_PATH / GTK_EXE_PREFIX / GIO_MODULE_DIR / LOCPATH into
# /snap/code/<rev>/, which makes any GTK binary spawned from this shell load
# snap's core20 glibc against the system one:
#   gnome-terminal.real: symbol lookup error: .../libpthread.so.0:
#   undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE
# Only affects processes started from inside the snap; a normal terminal is fine.
exec env \
  -u SNAP -u SNAP_NAME -u SNAP_REVISION -u SNAP_CONTEXT -u SNAP_EUID \
  -u SNAP_REAL_HOME -u SNAP_USER_COMMON -u SNAP_USER_DATA \
  -u SNAP_DATA -u SNAP_COMMON -u SNAP_LIBRARY_PATH -u SNAP_INSTANCE_NAME \
  -u GTK_EXE_PREFIX -u GTK_PATH -u GDK_PIXBUF_MODULE_FILE \
  -u GSETTINGS_SCHEMA_DIR -u LOCPATH -u GIO_MODULE_DIR \
  "$@"
