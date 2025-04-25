/* eslint-disable no-param-reassign */
/* eslint-disable no-underscore-dangle */
/* eslint-disable @typescript-eslint/naming-convention */
import axios from '@/utils/axios';
import { ip } from '@/utils/ip';


export interface IPayload {
  page: number,
  limit: number,
  cond: object,
}

class Users<T> {
  name: string;
  url: string;

  constructor({ name, url }: { name: string; url: string }) {
    this.name = name;
    this.url = url || name;
  }

  del = async (id: string) => {
    return axios.delete(`${ip}/user_info`, { params: { user_id: id } });
  };

  get = async (payload: IPayload) => {
    // console.log(payload);
    return axios.get(`${ip}/user_info`, { params: payload });
  };

  add = async (payload: T) => {
    Object.keys(payload).forEach((key) => {
      // payload[key] = payload[key];
      payload[key] = payload[key];
    });
    const formData = new FormData();
    Object.keys(payload).forEach((key) => {
      formData.append(key, payload[key]);
    });
    return axios.post(`${ip}/create_user_in_groups`, formData);
  };

  add2 = async (payload: T) => {
    Object.keys(payload).forEach((key) => {
      // payload[key] = payload[key];
      payload[key] = payload[key];
    });
    return axios.post(`${ip}/auth/${this.url}/`, payload);
  };

  upd = async (payload: T & { id: string | undefined }) => {
    const { id } = payload;
    payload.id = undefined;
    Object.keys(payload).forEach((key) => {
      payload[key] = payload[key];
    });
    const formData = new FormData();
    Object.keys(payload).forEach((key) => {
      formData.append(key, payload[key]);
    });
    return axios.put(`${ip}/create_user_in_groups`, formData);
  };
}

const users = new Users({ name: 'users', url: 'user_group' });

export default users;
