import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import {
  login as apiLogin,
  verifyLoginStep,
  getSession,
  logoutSession,
  setAuthToken,
  setCurrentAnalyst,
} from '../api';
import type { AuthUser, LoginResponse, Tier } from '../api/types';

/**
 * CLASSIFICATION-TIERED AUTHENTICATION
 *
 * Each KAVACH node guards one classification tier, and login strength scales
 * with it (checked by the backend, src/auth/tiered.py):
 *   low       Routine networks          password
 *   moderate  Sensitive networks        password + authenticator code (TOTP)
 *   high      Mission-critical networks password + TOTP + hardware security key
 * The tier also sets response authority: auto-execute, analyst confirms, or
 * two-person approval. Every login, failure and lockout is chained on the ledger.
 */

export interface User {
  name: string;
  email: string;
  role: string;
  initials: string;
  analystId?: string;
  username?: string;
  tier?: Tier;
  tierLabel?: string;
  zone?: string;
  responseRule?: AuthUser['responseRule'];
  responseLabel?: string;
  factors?: string[];
}

type PendingStep = Extract<LoginResponse, { status: 'step_required' }>;

interface AuthContextValue {
  isAuthenticated: boolean;
  user: User | null;
  /** Non-null while a moderate/high login is waiting for its next factor. */
  pendingStep: PendingStep | null;
  pendingUsername: string | null;
  /** Returns true when the session is complete, false when another factor is needed. */
  login: (username: string, password: string) => Promise<boolean>;
  submitStep: (proof: { code?: string; signature?: string }) => Promise<boolean>;
  cancelLogin: () => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const STORAGE_KEY = 'kavach_session';

function toUser(u: AuthUser): User {
  const initials = u.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
  return {
    name: u.name, email: u.analystId, role: u.role, initials, analystId: u.analystId, username: u.username,
    tier: u.tier, tierLabel: u.tierLabel, zone: u.zone, responseRule: u.responseRule, responseLabel: u.responseLabel,
    factors: u.factors,
  };
}

function readToken(): string | null {
  try {
    return sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function writeToken(token: string | null) {
  try {
    if (token) sessionStorage.setItem(STORAGE_KEY, token);
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    /* storage unavailable -- session lasts for this page only */
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [pendingStep, setPendingStep] = useState<PendingStep | null>(null);
  const [pendingUsername, setPendingUsername] = useState<string | null>(null);

  // Restore (and re-validate) the session on mount
  useEffect(() => {
    const token = readToken();
    if (!token) return;
    setAuthToken(token);
    getSession()
      .then(u => setUser(toUser(u)))
      .catch(() => {
        setAuthToken(null);
        writeToken(null);
      });
  }, []);

  // Every chained analyst action is attributed to the logged-in user
  useEffect(() => {
    setCurrentAnalyst(user?.analystId ?? user?.email);
  }, [user]);

  const handle = useCallback((res: LoginResponse): boolean => {
    if (res.status === 'ok') {
      setAuthToken(res.token);
      writeToken(res.token);
      setUser(toUser(res.user));
      setPendingStep(null);
      setPendingUsername(null);
      return true;
    }
    setPendingStep(res);
    return false;
  }, []);

  const login = async (username: string, password: string) => {
    const uname = username.includes('@') ? username.split('@')[0] : username;
    setPendingUsername(uname.trim().toLowerCase());
    return handle(await apiLogin(uname, password)); // throws on 401 / 423
  };

  const submitStep = async (proof: { code?: string; signature?: string }) => {
    if (!pendingStep) throw new Error('No login in progress');
    try {
      return handle(await verifyLoginStep(pendingStep.challengeId, proof));
    } catch (e) {
      // a failed factor consumes the challenge -- the analyst starts again
      setPendingStep(null);
      throw e;
    }
  };

  const cancelLogin = () => {
    setPendingStep(null);
    setPendingUsername(null);
  };

  const logout = () => {
    void logoutSession().catch(() => undefined);
    setAuthToken(null);
    writeToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{ isAuthenticated: user !== null, user, pendingStep, pendingUsername, login, submitStep, cancelLogin, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}

export const TIER_STYLE: Record<Tier, { label: string; chip: string; dot: string }> = {
  low: { label: 'Routine', chip: 'bg-risk-green-subtle text-risk-green border-risk-green/30', dot: 'bg-risk-green' },
  moderate: { label: 'Sensitive', chip: 'bg-risk-amber-subtle text-risk-amber border-risk-amber/30', dot: 'bg-risk-amber' },
  high: { label: 'Mission-critical', chip: 'bg-risk-red-subtle text-risk-red border-risk-red/30', dot: 'bg-risk-red' },
};
