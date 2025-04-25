// @ts-ignore
/* eslint-disable */
import axios from '@/utils/axios';
import { ip } from '@/utils/ip';
import { Login } from './typings';

export async function getInfo() {
  return axios.get(`${ip}/api/users/me`);
}

export async function putInfo(payload: Login.Profile) {
  return axios.put(`${ip}/user/me`, payload);
}

export async function register(payload: Login.RegisterPayload) {
  return axios.post(`${ip}/user/register`, payload);
}

export async function getInfoAdmin() {
  return axios.get(`${ip}/api/users/me`);
}

export async function login(payload: {
  username?: string;
  password?: string;
  deviceId: string;
  oneSignalId: string;
}) {
  return axios.post(`${ip}/auth/login/web`, payload);
}

export async function adminlogin(payload: { username?: string; password?: string }) {
  return axios.post(`${ip}/auth/login/web`, payload);
}

export async function changePassword(payload: { oldPassword: string; newPassword: string }) {
  return axios.post(`${ip}/user/me/change/password`, payload);
}

export async function forgetPassword(payload: { email: string }) {
  return axios.post(`${ip}/user/forget-password`, payload);
}

export async function resendEmail() {
  return axios.post(`${ip}/user/my/resend-active-token`);
}

export async function updateCCCD(payload: {
  cmtCccd: string;
  ngayCapCmtCccd: string;
  noiCapCmtCccd: string;
}) {
  return axios.put(`${ip}/user/me/cmt-cccd`, payload);
}

export async function updatePassword(idUser: string, payload: { password: string }) {
  return axios.put(`${ip}/user/${idUser}/password`);
}

export async function logout() {
  return axios.post(`${ip}/accounts/logout`);
}
