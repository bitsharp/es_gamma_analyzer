from jinja2.runtime import LoopContext, Macro, Markup, Namespace, TemplateNotFound, TemplateReference, TemplateRuntimeError, Undefined, escape, identity, internalcode, markup_join, missing, str_join
name = 'admin.html'

def root(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    l_0_admin_emails = resolve('admin_emails')
    l_0_mongo_enabled = resolve('mongo_enabled')
    l_0_sessions = resolve('sessions')
    try:
        t_1 = environment.filters['length']
    except KeyError:
        @internalcode
        def t_1(*unused):
            raise TemplateRuntimeError("No filter named 'length' found.")
    pass
    yield '<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8" />\n  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n  <title>Admin - Login Sessions</title>\n  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">\n  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">\n  <style>\n    body { background: #0d1b2a; color: #f8f9fa; min-height: 100vh; }\n    .navbar { background: linear-gradient(135deg, #1b263b 0%, #415a77 100%); border-bottom: 2px solid #778da9; }\n    .card { background: #1b263b; border: 2px solid #415a77; }\n    .card-header { background: #415a77; border-bottom: 2px solid #778da9; font-weight: 600; color: #ffffff; }\n    .table { --bs-table-bg: transparent; --bs-table-color: #f8f9fa; }\n    .table thead th { color: #e0e1dd; }\n    .text-muted { color: #adb5bd !important; }\n  </style>\n</head>\n<body>\n  <nav class="navbar mb-4">\n    <div class="container-fluid px-2 px-lg-3 d-flex align-items-center justify-content-between">\n      <div class="d-flex align-items-center gap-2">\n        <span class="fw-bold" style="font-size:1.2rem;">Admin</span>\n        <span class="text-muted">Login sessions</span>\n      </div>\n      <div class="d-flex align-items-center gap-2">\n        <a class="btn btn-sm btn-outline-light" href="/"><i class="bi bi-arrow-left"></i> Dashboard</a>\n        <a class="btn btn-sm btn-outline-light" href="/logout">Logout</a>\n      </div>\n    </div>\n  </nav>\n\n  <div class="container-fluid px-2 px-lg-3">\n    <div class="card">\n      <div class="card-header d-flex align-items-center justify-content-between">\n        <div><i class="bi bi-shield-lock me-2"></i>Login/Logout Events</div>\n        <div class="text-muted small">\n          '
    if (undefined(name='admin_emails') if l_0_admin_emails is missing else l_0_admin_emails):
        pass
        yield '\n            Admin allowlist: '
        yield escape((undefined(name='admin_emails') if l_0_admin_emails is missing else l_0_admin_emails))
        yield '\n          '
    else:
        pass
        yield '\n            Admin allowlist: (not set)\n          '
    yield '\n        </div>\n      </div>\n      <div class="card-body">\n        '
    if (not (undefined(name='mongo_enabled') if l_0_mongo_enabled is missing else l_0_mongo_enabled)):
        pass
        yield '\n          <div class="alert alert-warning mb-0">\n            MongoDB non configurato: impossibile mostrare i log.\n          </div>\n        '
    else:
        pass
        yield '\n          <div class="table-responsive">\n            <table class="table table-sm align-middle">\n              <thead>\n                <tr>\n                  <th>Time (UTC)</th>\n                  <th>Event</th>\n                  <th>User</th>\n                  <th>IP</th>\n                  <th class="text-nowrap">Session ID</th>\n                  <th>UA</th>\n                </tr>\n              </thead>\n              <tbody>\n                '
        for l_1_s in (undefined(name='sessions') if l_0_sessions is missing else l_0_sessions):
            _loop_vars = {}
            pass
            yield '\n                  <tr>\n                    <td class="text-nowrap">'
            yield escape((environment.getattr(l_1_s, 'created_at') or ''))
            yield '</td>\n                    <td class="text-nowrap">'
            yield escape((environment.getattr(l_1_s, 'event') or ''))
            if environment.getattr(l_1_s, 'provider'):
                pass
                yield ' <span class="text-muted">('
                yield escape(environment.getattr(l_1_s, 'provider'))
                yield ')</span>'
            yield '</td>\n                    <td class="text-nowrap">\n                      <div class="fw-semibold">'
            yield escape((environment.getattr(l_1_s, 'name') or ''))
            yield '</div>\n                      <div class="text-muted small">'
            yield escape((environment.getattr(l_1_s, 'email') or ''))
            yield '</div>\n                    </td>\n                    <td class="text-nowrap">'
            yield escape((environment.getattr(l_1_s, 'ip') or ''))
            yield '</td>\n                    <td class="text-nowrap"><code style="color:#e0e1dd;">'
            yield escape((environment.getattr(l_1_s, 'login_session_id') or ''))
            yield '</code></td>\n                    <td class="text-muted small" style="max-width:520px;">'
            yield escape((environment.getattr(l_1_s, 'user_agent') or ''))
            yield '</td>\n                  </tr>\n                '
        l_1_s = missing
        yield '\n\n                '
        if (t_1((undefined(name='sessions') if l_0_sessions is missing else l_0_sessions)) == 0):
            pass
            yield '\n                  <tr>\n                    <td colspan="6" class="text-muted">Nessun evento trovato.</td>\n                  </tr>\n                '
        yield '\n              </tbody>\n            </table>\n          </div>\n        '
    yield '\n      </div>\n    </div>\n  </div>\n\n  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>\n</body>\n</html>'

blocks = {}
debug_info = '38=21&39=24&46=30&64=36&66=40&67=42&69=49&70=51&72=53&73=55&74=57&78=61'