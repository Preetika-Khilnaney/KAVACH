import { useState } from 'react';
import { LogIn, Settings, LogOut, FileText } from 'lucide-react';
import { useAuth, TIER_STYLE } from '../contexts/AuthContext';
import { LoginModal } from './LoginModal';
import clsx from 'clsx';

interface LoginButtonProps {
  onOpenAccessibility: () => void;
  onOpenProvenance: () => void;
  /** 'lg' doubles the signed-out button's size -- used on the landing page. */
  size?: 'default' | 'lg';
}

export function LoginButton({ onOpenAccessibility, onOpenProvenance, size = 'default' }: LoginButtonProps) {
  const { isAuthenticated, user, logout } = useAuth();
  const [loginModalOpen, setLoginModalOpen] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  if (!isAuthenticated) {
    return (
      <>
        <button
          type="button"
          onClick={() => setLoginModalOpen(true)}
          data-interactive
          aria-label="Sign in"
          className={clsx(
            'flex items-center rounded-lg border border-border-default bg-surface hover:bg-canvas text-text-primary font-medium transition-colors',
            size === 'lg' ? 'gap-[9px] px-[14px] py-[7px] text-[14px] rounded-[9px]' : 'gap-2 px-3 py-1.5 text-xs',
          )}
        >
          <LogIn size={size === 'lg' ? 16 : 14} />
          <span className="hidden sm:inline">Sign in</span>
        </button>
        <LoginModal isOpen={loginModalOpen} onClose={() => setLoginModalOpen(false)} />
      </>
    );
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setDropdownOpen(!dropdownOpen)}
        onBlur={(e) => {
          // Close dropdown when focus leaves the button and dropdown
          if (!e.currentTarget.parentElement?.contains(e.relatedTarget as Node)) {
            setTimeout(() => setDropdownOpen(false), 100);
          }
        }}
        data-interactive
        aria-expanded={dropdownOpen}
        aria-haspopup="true"
        className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg border border-border-default bg-surface hover:bg-canvas text-text-primary transition-colors"
      >
        {/* Avatar with Initials */}
        <div className="w-6 h-6 rounded-full bg-accent-indigo text-white flex items-center justify-center text-[10px] font-heading font-bold">
          {user?.initials || 'U'}
        </div>
        <div className="hidden md:flex flex-col items-start min-w-0">
          <span className="text-xs font-medium leading-tight truncate max-w-[120px]">{user?.name}</span>
          {!user?.tier && <span className="text-[10px] text-text-tertiary leading-tight whitespace-nowrap">{user?.role}</span>}
        </div>
        {user?.tier && (
          <span
            className={clsx('hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-[9px] font-semibold uppercase tracking-wide whitespace-nowrap', TIER_STYLE[user.tier].chip)}
            title={user.zone}
          >
            <span className={clsx('w-1.5 h-1.5 rounded-full', TIER_STYLE[user.tier].dot)} aria-hidden="true" />
            {TIER_STYLE[user.tier].label}
          </span>
        )}
      </button>

      {/* Dropdown Menu */}
      {dropdownOpen && (
        <div
          className="absolute top-full right-0 mt-2 w-72 bg-surface rounded-lg border border-border-default shadow-panel py-1 z-50"
          role="menu"
        >
          {/* User Info Header */}
          <div className="px-3 py-2 border-b border-border-default">
            <p className="text-xs font-medium text-text-primary">{user?.name}</p>
            <p className="text-[11px] text-text-secondary">{user?.email}</p>
            {user?.tier && (
              <div className="mt-1.5 space-y-0.5">
                <p className="text-[10px] text-text-tertiary">Zone: <span className="text-text-secondary">{user.zone}</span></p>
                <p className="text-[10px] text-text-tertiary">Signed in with: <span className="text-text-secondary">{user.factors?.map(f => f === 'totp' ? 'authenticator' : f === 'hardware_key' ? 'security key' : f).join(' + ')}</span></p>
                <p className="text-[10px] text-text-tertiary">Response rule: <span className="text-text-secondary">{user.responseLabel}</span></p>
              </div>
            )}
          </div>

          {/* Menu Items */}
          <button
            type="button"
            onClick={() => {
              setDropdownOpen(false);
              onOpenProvenance();
            }}
            data-interactive
            role="menuitem"
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-text-primary hover:bg-canvas transition-colors"
          >
            <FileText size={14} className="text-accent-teal" />
            Model Provenance
          </button>

          <button
            type="button"
            onClick={() => {
              setDropdownOpen(false);
              onOpenAccessibility();
            }}
            data-interactive
            role="menuitem"
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-text-primary hover:bg-canvas transition-colors"
          >
            <Settings size={14} className="text-accent-indigo" />
            Accessibility Settings
          </button>

          <div className="h-px bg-border-default my-1" />

          <button
            type="button"
            onClick={() => {
              setDropdownOpen(false);
              logout();
            }}
            data-interactive
            role="menuitem"
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-risk-red hover:bg-risk-red-subtle transition-colors"
          >
            <LogOut size={14} />
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
