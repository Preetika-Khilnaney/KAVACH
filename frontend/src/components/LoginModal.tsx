import { useRef, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';
import { createPortal } from 'react-dom';
import {
  X, KeyRound, Eye, EyeOff, Smartphone, Usb, Check, Loader2, Lock, ChevronRight, AlertTriangle, Info,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { getDemoAuthenticatorCode, signWithKeyFile } from '../api';
import type { LoginFactor, Tier } from '../api/types';
import { useFocusTrap } from '../hooks/useFocusTrap';
import loginBackground from '../assets/login-background.png';
import clsx from 'clsx';

interface LoginModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const NAVY = '#0d2a52';

const FACTOR_META: Record<LoginFactor, { label: string; icon: typeof KeyRound }> = {
  password: { label: 'Password', icon: KeyRound },
  totp: { label: 'Authenticator', icon: Smartphone },
  hardware_key: { label: 'Security key', icon: Usb },
};

const TIERS: { tier: Tier; name: string; zone: string; factors: LoginFactor[]; rule: string; bar: string; text: string; ring: string }[] = [
  { tier: 'low', name: 'Routine', zone: 'Office · admin', factors: ['password'], rule: 'Auto-response',
    bar: 'bg-risk-green', text: 'text-risk-green', ring: 'ring-risk-green/50 bg-risk-green-subtle' },
  { tier: 'moderate', name: 'Sensitive', zone: 'Finance · HR', factors: ['password', 'totp'], rule: 'Analyst confirms',
    bar: 'bg-risk-amber', text: 'text-risk-amber', ring: 'ring-risk-amber/50 bg-risk-amber-subtle' },
  { tier: 'high', name: 'Mission-critical', zone: 'Classified · grid', factors: ['password', 'totp', 'hardware_key'], rule: 'Two-person rule',
    bar: 'bg-risk-red', text: 'text-risk-red', ring: 'ring-risk-red/50 bg-risk-red-subtle' },
];

function errorText(error: unknown): string {
  const msg = error instanceof Error ? error.message : String(error);
  const m = msg.match(/: \d{3} [^:]*: (.*)$/);
  if (m) return m[1];
  if (msg.includes('Failed to fetch')) return 'Cannot reach the KAVACH node. Check that the backend is running on port 8000.';
  return msg;
}

function Field({ label, htmlFor, children }: { label: string; htmlFor: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="block text-[11px] font-semibold uppercase tracking-[0.08em] text-text-secondary">
        {label}
      </label>
      {children}
    </div>
  );
}

const INPUT =
  'w-full h-11 px-3.5 rounded-lg border border-border-default bg-white text-sm text-text-primary placeholder:text-text-tertiary ' +
  'focus:outline-none focus:border-[#16407a] focus:ring-4 focus:ring-[#16407a]/10 transition-shadow';

function PrimaryButton({ loading, children, type = 'submit', onClick }: { loading: boolean; children: ReactNode; type?: 'submit' | 'button'; onClick?: () => void }) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={loading}
      data-interactive
      style={{ backgroundColor: NAVY }}
      className={clsx(
        'w-full h-11 rounded-lg font-heading font-semibold text-[13px] tracking-wide text-white flex items-center justify-center gap-2',
        'shadow-sm transition-all hover:brightness-125 focus:outline-none focus-visible:ring-4 focus-visible:ring-[#16407a]/30',
        loading && 'opacity-70 cursor-not-allowed',
      )}
    >
      {loading ? <Loader2 size={16} className="animate-spin" /> : null}
      {children}
    </button>
  );
}

const KAVACH_LETTERS = 'Kavach'.split('');
const FLY_IN_DURATION = 1.1;

/** Right-panel wordmark: flies in rotating from the top-right corner, settles
 * center, then each letter hops up in sequence and comes to rest. */
function AnimatedKavachWordmark() {
  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none" aria-hidden="true">
      <motion.div
        initial={{ x: 140, y: -120, rotate: -70, opacity: 0 }}
        animate={{ x: 0, y: 0, rotate: 0, opacity: 1 }}
        transition={{ duration: FLY_IN_DURATION, ease: [0.16, 1, 0.3, 1] }}
        className="flex"
      >
        {KAVACH_LETTERS.map((letter, idx) => (
          <motion.span
            key={idx}
            initial={{ y: 0 }}
            animate={{ y: [0, -16, 0] }}
            transition={{
              duration: 0.5,
              delay: FLY_IN_DURATION + idx * 0.1,
              times: [0, 0.5, 1],
              ease: 'easeOut',
            }}
            className="font-heading font-bold text-white text-6xl tracking-tight inline-block"
          >
            {letter}
          </motion.span>
        ))}
      </motion.div>
    </div>
  );
}

/**
 * CLASSIFICATION-TIERED LOGIN
 *
 * Step 1 password for everyone; the backend then asks for the factors the
 * user's tier requires: an authenticator (TOTP) code for Sensitive and
 * Mission-critical zones, and a hardware security key challenge for
 * Mission-critical. See src/auth/tiered.py.
 */
export function LoginModal({ isOpen, onClose }: LoginModalProps) {
  const { login, submitStep, cancelLogin, pendingStep, pendingUsername } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [code, setCode] = useState('');
  const [demoCode, setDemoCode] = useState<{ code: string; validFor: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);
  const keyInput = useRef<HTMLInputElement>(null);

  const containerRef = useFocusTrap(isOpen);

  const factors: LoginFactor[] = pendingStep?.factors ?? ['password'];
  const done: LoginFactor[] = pendingStep?.completed ?? [];
  const current: LoginFactor = pendingStep?.step ?? 'password';
  const tier: Tier | null = pendingStep?.tier ?? null;
  const tierInfo = TIERS.find(t => t.tier === tier);

  const reset = () => {
    setUsername('');
    setPassword('');
    setCode('');
    setDemoCode(null);
    setLoginError(null);
  };

  const run = async (fn: () => Promise<boolean>) => {
    setLoading(true);
    setLoginError(null);
    try {
      if (await fn()) {
        reset();
        onClose();
        navigate('/dashboard');
      }
    } catch (error) {
      setLoginError(errorText(error));
      setCode('');
      setDemoCode(null);
    } finally {
      setLoading(false);
    }
  };

  const handlePassword = (e: FormEvent) => {
    e.preventDefault();
    void run(() => login(username, password));
  };

  const handleCode = (e: FormEvent) => {
    e.preventDefault();
    void run(() => submitStep({ code }));
  };

  const handleKeyFile = async (file: File | undefined) => {
    if (!file || !pendingStep?.nonce) return;
    const text = await file.text();
    void run(async () => {
      const { signature } = await signWithKeyFile(text, pendingStep.nonce!);
      return submitStep({ signature });
    });
    if (keyInput.current) keyInput.current.value = '';
  };

  const showDemoAuthenticator = async () => {
    if (!pendingUsername) return;
    try {
      setDemoCode(await getDemoAuthenticatorCode(pendingUsername));
    } catch (error) {
      setLoginError(errorText(error));
    }
  };

  const handleClose = () => {
    cancelLogin();
    reset();
    onClose();
  };

  if (!isOpen) return null;

  const heading = current === 'password' ? 'Sign in' : current === 'totp' ? 'Verify your identity' : 'Present your security key';
  const sub =
    current === 'password'
      ? 'Access the KAVACH console for your security zone.'
      : current === 'totp'
        ? `Enter the 6-digit code from the authenticator registered to ${pendingUsername}.`
        : `Mission-critical zone. Insert the security key registered to ${pendingUsername}.`;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={handleClose}>
      <div className="absolute inset-0 bg-[#050e1f]/60 backdrop-blur-sm" aria-hidden="true" />

      <motion.div
        ref={containerRef}
        role="dialog"
        aria-label="Sign in to KAVACH"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
        initial={{ opacity: 0, y: 16, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 w-full max-w-[980px] max-h-[94vh] overflow-hidden rounded-2xl shadow-[0_30px_80px_-20px_rgba(5,14,31,.6)] border border-white/10 flex flex-col bg-surface"
      >
        {/* Classification banner */}
        <div className="flex items-center justify-center gap-2 px-4 py-1.5 text-[10px] font-semibold uppercase tracking-[0.2em] text-[#5c4308] bg-[#fdf1d6] border-b border-[#ecd28f]">
          <Lock size={11} aria-hidden="true" /> Restricted system · Authorised personnel only · All activity is logged
        </div>

        <div className="flex min-h-0 flex-1">
          {/* Left: form */}
          <div className="relative w-full sm:w-[440px] shrink-0 overflow-y-auto px-9 pt-8 pb-7 flex flex-col">
            <button
              type="button"
              onClick={handleClose}
              aria-label="Close sign-in dialog"
              data-interactive
              className="absolute top-4 right-4 p-1.5 rounded-md text-text-tertiary hover:text-text-primary hover:bg-canvas transition-colors"
            >
              <X size={18} />
            </button>

            {/* Mobile brand */}
            <div className="sm:hidden flex items-center gap-2 mb-5">
              <img src="/KAVACH_logo.png" alt="" className="w-8 h-8 object-contain" />
              <span className="font-heading font-bold tracking-[0.15em] text-text-primary">KAVACH</span>
            </div>

            <AnimatePresence mode="wait">
              <motion.div key={current} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -10 }} transition={{ duration: 0.18 }}>
                <h2 className="font-heading font-bold text-[22px] text-text-primary tracking-tight">{heading}</h2>
                <p className="text-[13px] text-text-secondary mt-1 leading-snug">{sub}</p>
              </motion.div>
            </AnimatePresence>

            {/* Zone tiers */}
            <div className="mt-6">
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-secondary mb-2">Security zones</p>
              <div className="grid grid-cols-3 gap-2" role="list" aria-label="Security zones and their sign-in requirements">
                {TIERS.map(t => {
                  const active = tier === t.tier;
                  return (
                    <div
                      key={t.tier}
                      role="listitem"
                      aria-current={active ? 'true' : undefined}
                      className={clsx(
                        'relative overflow-hidden rounded-lg border border-border-default px-2.5 pt-3 pb-2.5 transition-all',
                        active && `ring-2 ${t.ring} border-transparent`,
                        tier && !active && 'opacity-40',
                      )}
                    >
                      <span className={clsx('absolute top-0 left-0 right-0 h-[3px]', t.bar)} aria-hidden="true" />
                      <p className={clsx('text-[11.5px] font-bold leading-tight', t.text)}>{t.name}</p>
                      <p className="text-[10px] text-text-tertiary mt-0.5">{t.zone}</p>
                      <div className="flex items-center gap-1 mt-2 text-text-secondary">
                        {t.factors.map(f => {
                          const Icon = FACTOR_META[f].icon;
                          return <Icon key={f} size={12} aria-label={FACTOR_META[f].label} />;
                        })}
                      </div>
                      <p className="text-[9.5px] text-text-tertiary mt-1">{t.rule}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Step progress */}
            {pendingStep && (
              <ol className="mt-5 flex items-center" aria-label="Sign-in progress">
                {factors.map((f, i) => {
                  const Icon = FACTOR_META[f].icon;
                  const isDone = done.includes(f);
                  const isCur = f === current;
                  return (
                    <li key={f} className="flex items-center flex-1 last:flex-none">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={clsx(
                            'w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold border',
                            isDone ? 'bg-risk-green border-risk-green text-white' : isCur ? 'border-[#16407a] text-[#16407a] bg-[#16407a]/10' : 'border-border-default text-text-tertiary',
                          )}
                        >
                          {isDone ? <Check size={12} /> : <Icon size={11} />}
                        </span>
                        <span className={clsx('text-[11px] font-medium', isCur ? 'text-text-primary' : 'text-text-tertiary')}>{FACTOR_META[f].label}</span>
                      </div>
                      {i < factors.length - 1 && <span className={clsx('flex-1 h-px mx-2', isDone ? 'bg-risk-green' : 'bg-border-default')} />}
                    </li>
                  );
                })}
              </ol>
            )}

            <div className="mt-6">
              {/* Step 1: password */}
              {current === 'password' && (
                <form onSubmit={handlePassword} className="space-y-4">
                  <Field label="Username" htmlFor="username">
                    <input id="username" type="text" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)}
                      placeholder="e.g. officer.a" required autoFocus className={INPUT} />
                  </Field>
                  <Field label="Password" htmlFor="password">
                    <div className="relative">
                      <input id="password" type={showPassword ? 'text' : 'password'} autoComplete="current-password" value={password}
                        onChange={(e) => setPassword(e.target.value)} placeholder="Enter your password" required className={clsx(INPUT, 'pr-10')} />
                      <button type="button" onClick={() => setShowPassword(v => !v)} data-interactive aria-label={showPassword ? 'Hide password' : 'Show password'}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-text-tertiary hover:text-text-primary">
                        {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                      </button>
                    </div>
                  </Field>
                  <PrimaryButton loading={loading}>
                    Continue <ChevronRight size={16} />
                  </PrimaryButton>
                </form>
              )}

              {/* Step 2: authenticator */}
              {current === 'totp' && (
                <form onSubmit={handleCode} className="space-y-4">
                  <Field label="Authenticator code" htmlFor="otp">
                    <input id="otp" type="text" inputMode="numeric" autoComplete="one-time-code" pattern="[0-9]{6}" maxLength={6} value={code}
                      onChange={(e) => setCode(e.target.value.replace(/\D/g, ''))} placeholder="000000" required autoFocus
                      className={clsx(INPUT, 'h-14 text-center font-mono text-2xl tracking-[0.5em] placeholder:tracking-[0.5em]')} />
                  </Field>
                  <PrimaryButton loading={loading}>
                    Verify code <ChevronRight size={16} />
                  </PrimaryButton>
                  <div className="flex items-center justify-between rounded-lg bg-canvas border border-border-default px-3 py-2">
                    <span className="flex items-center gap-1.5 text-[11px] text-text-tertiary">
                      <Smartphone size={12} /> Demo authenticator
                    </span>
                    {demoCode ? (
                      <span className="text-[11px] text-text-secondary">
                        <strong className="font-mono text-sm text-text-primary tracking-widest">{demoCode.code}</strong> · {demoCode.validFor}s
                      </span>
                    ) : (
                      <button type="button" data-interactive onClick={showDemoAuthenticator} className="text-[11px] font-medium text-[#16407a] hover:underline">
                        Show code
                      </button>
                    )}
                  </div>
                </form>
              )}

              {/* Step 3: hardware key */}
              {current === 'hardware_key' && (
                <div className="space-y-4">
                  <div className="rounded-lg border border-border-default bg-canvas px-3.5 py-3">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-secondary">Challenge</span>
                      {pendingStep?.keyId && <span className="font-mono text-[10px] text-text-tertiary">key {pendingStep.keyId}</span>}
                    </div>
                    <p className="font-mono text-[11px] text-text-primary break-all mt-1">{pendingStep?.nonce}</p>
                  </div>
                  <input ref={keyInput} type="file" accept=".kavachkey,application/json" className="hidden"
                    onChange={(e) => void handleKeyFile(e.target.files?.[0])} />
                  <PrimaryButton type="button" loading={loading} onClick={() => keyInput.current?.click()}>
                    {!loading && <Usb size={16} />} Insert security key
                  </PrimaryButton>
                  <p className="text-[11px] text-text-tertiary leading-relaxed">
                    The key signs the challenge on this machine; its secret is never transmitted. Prototype keys are in{' '}
                    <span className="font-mono">data/auth/keys/</span>.
                  </p>
                </div>
              )}
            </div>

            {loginError && (
              <div role="alert" className="mt-4 flex items-start gap-2 rounded-lg border border-risk-red/30 bg-risk-red-subtle px-3 py-2.5 text-[12px] text-risk-red">
                <AlertTriangle size={14} className="shrink-0 mt-px" /> <span>{loginError}</span>
              </div>
            )}

            {tierInfo && (
              <p className="mt-4 text-[11px] text-text-tertiary">
                Signing in to a <strong className={tierInfo.text}>{tierInfo.name}</strong> zone · {factors.length} factor{factors.length > 1 ? 's' : ''} required
              </p>
            )}

            <div className="mt-auto pt-6 space-y-3">
              {pendingStep ? (
                <button type="button" data-interactive onClick={() => { cancelLogin(); setLoginError(null); }}
                  className="text-[12px] font-medium text-[#16407a] hover:underline">
                  ← Use a different account
                </button>
              ) : (
                <details className="group rounded-lg border border-border-default px-3 py-2 text-[11px] text-text-secondary">
                  <summary className="cursor-pointer list-none flex items-center gap-1.5 font-medium text-text-secondary">
                    <Info size={12} /> Demo credentials
                    <ChevronRight size={12} className="ml-auto transition-transform group-open:rotate-90" />
                  </summary>
                  <div className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px]">
                    <span>l1.analyst</span><span className="text-risk-green font-sans">Routine</span>
                    <span>l2.responder</span><span className="text-risk-amber font-sans">Sensitive</span>
                    <span>officer.a / .b</span><span className="text-risk-red font-sans">Mission-critical</span>
                    <span className="text-text-tertiary font-sans">Password</span><span>kavach2026</span>
                  </div>
                </details>
              )}
              <p className="text-[10.5px] leading-relaxed text-text-tertiary border-t border-border-default pt-3">
                Unauthorised access or attempted access is an offence under Sections 43 and 66 of the Information Technology Act, 2000.
                Every attempt is recorded on the tamper-evident ledger; five failures lock the account for five minutes.
              </p>
            </div>
          </div>

          {/* Right: Animated panel */}
          <div
            className="relative hidden sm:block flex-1 bg-cover bg-center overflow-hidden"
            style={{ backgroundImage: `url(${loginBackground})` }}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-[#0F0B2E]/80 via-[#1B1464]/60 to-[#2D1B69]/70" aria-hidden="true" />
            <AnimatedKavachWordmark />
          </div>
        </div>
      </motion.div>
    </div>,
    document.body,
  );
}
